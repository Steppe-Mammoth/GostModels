"""
ElasticModel is a base class for working with partial (projection) documents from a database.

It allows you to create objects from truncated dicts, ignoring missing required fields, but:
- will throw an error when accessing an unloaded field (NotLoadedFieldError);
- validates existing values typically via TypeAdapter (email, datetime, enum, …);
- recursively builds nested models (which also inherit ElasticModel);
- puts extra keys in .elastic_extra (without validation);
- supports two validation modes: deep and shallow.

ElasticModel — базовий клас для роботи з частковими (проекційними) документами з бази даних.

Дозволяє створювати об'єкти з урізаних dict'ів, ігноруючи відсутні обов'язкові поля, але:
- кидає помилку при доступі до незавантаженого поля (NotLoadedFieldError);
- валідує наявні значення зазвичай через TypeAdapter (email, datetime, enum, …);
- рекурсивно будує вкладені моделі (які також наслідують ElasticModel);
- поміщає зайві ключі в .extra (без валідації);
- підтримує два режими валідації: глибокий та поверхневий.
"""

# pydantic опис методів:
# .model_extra - Невідомі моделі поля (зберігаються в окремому місці в залежності від `ConfigDict.extra`), до цих полів можна отримати доступ як і до всіх інших полів через instance.key
# .__dict__ - Відомі моделі заповнені поля (Юзерські + Дефолтні)
# .model_dump() - `.__dict__` + `.model_extra`. Відображає всі поля які доступні через "."
# .model_fields_set / __pydantic_fields_set__ - Відомі моделі поля які встановлені виключно юзером (без дефолтів, та без `.model_extra`). 
#   (Увага!. Масив не реагує на видалення полів (якщо юзер видалить поле, воно все ще залишиться тут), зате реагує на додавання полів)
# 

# elastic_model опис методів:
# .elastic_loaded_fields - Відомі та не відомі моделі юзерські поля (без дефолтів, + `.model_extra`, які прямо доступні через "."
#   - Завжди показує актуальне значення завантажених полів, (По суті комбінація `model_extra + .model_fields_set`, але навідміну від `.model_fields_set` точніше за рахунок видалення (актуальності) полів) 
#   - Якщо юзер видалить завантажене поле - воно також зникне з `.elastic_loaded_fields` (навідміну від `model_fields_set`), 

# .elastic_extra - невідомі моделі поля, завжди зберігаються незважаючи на `ConfigDict.extra`, 
#   - Дозволяє зберігати ці поля навіть при `ConfigDict.extra="ignore`,
#     (`ConfigDict.extra="ignore` дозволяє не засмічувати модель лишніми полями, як при `ConfigDict.extra="always`)
# .


from __future__ import annotations


from typing import Any, Annotated, Mapping, Self, Union, get_args, get_origin, get_type_hints
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, PrivateAttr, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

import logging
logger = logging.getLogger(__name__)

# =========================
#   Helper Entities 
# =========================

_SYSTEM_ATTRS = (
    'model_config',
    'model_fields',
    '__dict__',
    '__class__',
    '__fields_set__',
    '__pydantic_fields_set__',
    # Публічні утиліти обгортки:
    'elastic_is_loaded',
    'elastic_get_model_fields',
    'elastic_is_valid',
    'elastic_get_validated_model',
    'elastic_extra',
)


def _issubclass_safe(tp: Any, base: type) -> bool:
    """
    Safe issubclass check for cases where tp might not be a class (e.g., typing constructs).
    Used only as a safe check. The function guarantees False instead of an error.
    
    Безпечна перевірка issubclass для випадків, коли tp може бути не-класом (наприклад, typing-конструкції).
    Використовується лише як безпечна перевірка. Функція гарантує False замість помилки.
    """
    try:
        return isinstance(tp, type) and issubclass(tp, base)
    except TypeError:
        return False


@lru_cache(maxsize=256)
def _raw_annotations_map(cls: type) -> dict[str, Any]:
    # Returns annotations with Annotated/Field(...) inside
    return get_type_hints(cls, include_extras=True)


@lru_cache(maxsize=512)
def _adapter_hashable(annotation: Any) -> TypeAdapter:
    return TypeAdapter(annotation)

def _adapter(annotation: Any) -> TypeAdapter:
    """
    Cached factory method for TypeAdapter(annotation).
    TypeAdapter in Pydantic v2 is a validator/coercer for type-hint without creating BaseModel.
    Creating an adapter is not free, so cache significantly reduces overhead.
    
    Кешований фабричний метод для TypeAdapter(annotation).
    TypeAdapter в Pydantic v2 — це валідатор/коерсер за type-hint'ом без створення BaseModel.
    Створення адаптера не безкоштовне, тому кеш суттєво зменшує накладні витрати.

    - Purpose / Мета: 
        - Get cached pydantic.TypeAdapter(annotation) — mechanism that validates/coerces values by type-hint without creating a model.
        - Отримати кешований pydantic.TypeAdapter(annotation) — механізм, який валідовує/коерсить значення за type-hint’ом без створення моделі.
    - Why cache / Чому кеш:
        - Significantly speeds up repeated validations of identical types.
        - Значно пришвидшує повторні валідації однакових типів.
    - Notes / Нотатки:
        - Cache key is the annotation itself. If you pass the same type object (e.g., EmailStr, list[int], MyModel), the adapter is taken from cache.
        - Ключем кешу є сам annotation. Якщо ти передаєш той самий об'єкт типу (наприклад, EmailStr, list[int], MyModel), адаптер береться з кешу.
    """
    # Annotated[..., FieldInfo] might be unhashable (because FieldInfo is inside).
    try:
        return _adapter_hashable(annotation)  # Try cache by hash
    except TypeError:
        # e.g. Annotated[..., Field(...)] often not hashable → this is expected
        logger.warning(
            "ElasticModel: unhashable annotation for adapter cache; using non-cached adapter: %r",
            annotation,
        )
        return TypeAdapter(annotation)      # fallback without cache 
    except Exception:
        logger.exception(
            "ElasticModel: unexpected error creating TypeAdapter for %r; falling back to non-cached instance",
            annotation,
        )
        return TypeAdapter(annotation)        # fallback without cache


def _strip_annot(annotation: Any) -> Any:
    """
    Removes only the outer wrapper Annotated[..., meta], returning the base type for structure analysis (Union/list/dict/tuple/class).
    IMPORTANT: for actual validation, use the ORIGINAL annotation (with metadata).
    
    Знімає лише зовнішню обгортку Annotated[..., meta], повертаючи базовий тип для аналізу структури (Union/list/dict/tuple/клас).
    ВАЖЛИВО: для самої валідації використовуйте ОРИГІНАЛЬНУ анотацію (з метаданими).
    """

    origin = get_origin(annotation)
    if origin is Annotated:
        return get_args(annotation)[0]
    else:
        return annotation


def _build_validation_payload(model: BaseModel, recursive: bool) -> dict[str, Any]:
    """
    Forms payload for validation via BaseModel.model_validate(...).
    
    Parameters:
    - recursive:
        - Якщо True - Повна серіалізація. pydantic створить, та провалідує всі рівні
        - Якщо False - Поверхнева серіалізація. Вкладені моделі залишаються інстансами, pydantic інстани залишає як є

    Notes:
        - If `ConfigDict.revalidate_instances != 'never'`, then even nested model instances will be revalidated.
        - Якщо `ConfigDict.revalidate_instances != 'never'`, то навіть інстанси вкладених моделей будуть перевалідовані.
    """

    # Full serialization
    if recursive:
        # Сериалізуємо все, pydantic все створить з нуля
        data = model.model_dump(exclude_unset=False)  # Нехай дампить дефолтні значення. Інакше прийдеть обробляти else блок для однорідності логіки !todo 
        return data
    
    # Shallow serialization (Поверхнева).
    else:
        # Всі вкладені інстанси залишаємо як є, pydantic їх теж залишить як є
        model_dict = dict(object.__getattribute__(model, "__dict__"))  # Відомі моделі поля (Юзерські + Дефолт)
        model_extra = getattr(model, "model_extra", None) or {} # Невідомі моделі юзерські поля
        data = model_dict | model_extra 
        
        return data


# =========================
#   Main Class 
# =========================


class ElasticModel(BaseModel):
    """
    A wrapper class around 'BaseModel' that allows constructing model objects with various data sets:
        - Without requiring all fields
        - With extra fields important for context, which will be saved in `.extra`, for access to them

    This will allow us to create models with limited/excessive data obtained from external resources (e.g., DB)
        
    Mechanics:
    - Create instances from projection documents via elastic_create().
    - Access to unloaded fields throws NotLoadedFieldError.
    - Extra keys available in .extra (without validation).
    - Nested ElasticModels work recursively with the same semantics.
    - For "full" validation and running class validators, use to_validated().
    """

    model_config = ConfigDict(
        extra='ignore',                 # Ignore extra keys at model level, but save them in .elastic_extra for manual access. / Ігноруємо лишні ключі на рівні моделі, але зберігаємо їх в .elastic_extra для ручного доступу.
        populate_by_name=True,          # Allows substituting data by both alias and field name
        revalidate_instances='never'    # Don't validate nested object if it's already a BaseModel instance / Не валідуємо вкладений об'єкт, якщо він вже є інстансом BaseModel
    )

    # Private state-carrying fields
    _elastic_finished: bool = PrivateAttr(default=False)
    _elastic_extra: dict[str, Any] = PrivateAttr(default_factory=dict)  # All unknown model fields are stored here (without validation)
    _elastic_loaded_fields: set[str] = PrivateAttr(default_factory=set) # All loaded model fields are stored here


    @classmethod
    def elastic_create(
        cls,
        data: dict[str, Any],
        validate: bool = True,
        strict_validate: bool = False,
        defaults: bool = False
    ) -> Self:
        """
        Build a partial class instance from a truncated dict.

        :param data: `dict` (may contain partial fields and "extra" keys which will go to `.extra`).
        :param validate: 
            - If True (default) - all values from `data` will be validated and converted according to their type annotations in the model.
            - If False - values are accepted "as is" (only for trusted flows).
        :param apply_defaults:
            - If True - `default`/`default_factory` are substituted for missing fields.
            - If False - missing fields are not created; accessing them will throw `NotLoadedFieldError`.
        
        :return Class instance with:
            - Only those fields set that are in `data` (And default values if `apply_defaults == True`)

            - `.elastic_extra` - dictionary with all unknown keys at model level (without validation), 
            (If more fields are passed than described in the model, they will be in this dictionary);

            - `._elastic_loaded_fields` - list (set) of fields that were actually set.

        :param strict_validate:
            - Якщо True значення не буде коерсить pydantic'ом перед валідацією

        Note:
        - Class validators (`field_validator`/`model_validator`) are NOT run at this stage. Run them via `to_validated()` or regular `model_validate()`.
        """
        model_fields = cls.model_fields  # Всі поля задекларовані в моделі
        raw_annotations = _raw_annotations_map(cls)
        
        # Словник аліасів {"_id": "id"}
        alias_to_name = {f.alias or n: n for n, f in model_fields.items()}  # Support alias: e.g., "_id" -> "id"
        
        elastic_data: dict[str, Any] = {}
        elastic_extra: dict[str, Any] = {}

        all_loaded_fields = []  # Всі поля які передав юзер, з врахуванням алісів
        elastic_loaded_fields = set()  # Поля юзера які були збережні в модель, та не відсіяні через її налаштування (ConfigDict.extra)

        # Чи зберігати невідомі моделі поля в elastic_extra
        unknown_fields_to_elastic_extra = True  # !todo Надати можливість змінювати цей параметр

        for raw_field_name, raw_field_value in data.items():
            instance_field_name = alias_to_name.get(raw_field_name, raw_field_name)  # Правильна назва поля, з врахуванням аліасу
            
            all_loaded_fields.append(instance_field_name)

            if instance_field_name in model_fields:
                annotation = raw_annotations.get(instance_field_name, model_fields[instance_field_name].annotation)
                elastic_data[instance_field_name] = cls._elastic_coerce_value(annotation=annotation, value=raw_field_value, validate=validate, strict_validate=strict_validate)
            else:
                if unknown_fields_to_elastic_extra:
                    # Додатоков зберігаємо в elastic_extra
                    elastic_extra[raw_field_name] = raw_field_value

                # Якщо поле невідоме - все рівно додаємо його до elastic_data,
                # Делегуємо їх долю в .model_construct. Він сам вирішить в залежності від ConfigDict.extra
                elastic_data[instance_field_name] = raw_field_value

        # Створення моделі
        instance: BaseModel = cls.model_construct(**elastic_data)
        object.__setattr__(instance, "_elastic_finished", False)

        # Проходимось по всім полям нової моделі
        #   - Формуємо список `elastic_loaded_fields`
        #   - Видаляємо заяйві поля (`defaults==False`)
        instance_dump = instance.model_dump() # Всі поля прямо доступні через "." (Юзерскі+Дефолтні+Extra)
        for instance_field_name, _ in instance_dump.items():
            
            # Юзерське поле. (логіка для прямих Юзерських + Extra полів)
            if instance_field_name in all_loaded_fields:
                elastic_loaded_fields.add(instance_field_name)  # `elastic_loaded_fields` збереже це завантажене юзером поле, яке прямо доступне через "."
            
            # Дефолтне поле (`.model_construct` встановив дефолти)
            else:
                # Юзер хоче видалити дефолтні поля, щоб не путати їх з реально завантаженими полями
                if not defaults:
                    # Видаляємо дефолтне поле з __dict__ 
                    delattr(instance, instance_field_name)

        
        # Save service information.
        object.__setattr__(instance, "_elastic_extra", elastic_extra)
        object.__setattr__(instance, "_elastic_loaded_fields", elastic_loaded_fields)
        object.__setattr__(instance, "_elastic_finished", True)  # Ставимо мітку, що ми успішно звершили створення інстансу
        return instance

    @property
    def elastic_extra(self) -> dict[str, Any]:
        """
        All unknown model fields are stored here .
        """
        return self._elastic_extra
    
    @property
    def elastic_loaded_fields(self) -> set[str]:
        return self._elastic_loaded_fields

    def elastic_is_loaded(self, name: str) -> bool:
        """
        Checks whether the field was set during `elastic_create`.
        """
        return name in self.elastic_loaded_fields

        # try:
        #     loaded = object.__getattribute__(self, "_elastic_loaded_fields")
        # except AttributeError:
        #     logger.warning(
        #         "ElasticModel: '_elastic_loaded_fields' not initialized yet on %s while checking is_loaded('%s')",
        #         type(self).__name__, name,
        #     )
        #     return False
        # return name in loaded
    
    def elastic_is_valid(self, recursive: bool = True) -> tuple[bool, list[str]]:
        """
        Checks validity without returning a new instance.
        Returns (True/False, bad_paths: List[str]), where bad_paths — 'a.b[2].c' etc.
        """
        payload = _build_validation_payload(model=self, recursive=recursive)
        
        try:
            # _adapter(self.__class__).validate_python(payload)
            self.__class__.model_validate(payload)
            return True, []
        except ValidationError as e:
            # create a list of invalid fields
            paths: list[str] = []
            for err in e.errors():
                loc = err.get("loc", ())
                parts: list[str] = []
                for p in loc:
                    if isinstance(p, int):
                        if parts:
                            parts[-1] = f"{parts[-1]}[{p}]"
                        else:
                            parts.append(f"[{p}]")
                    else:
                        parts.append(str(p))
                paths.append(".".join(parts))
            return False, paths

    def elastic_get_validated_model(self, recursive: bool = True) -> Self:
        """
        Full validation of current model state:
        - If validation is successful - returns a new class instance
        - If validation is unsuccessful - throws ValidationError.

        (If you just want to know if this object is valid — use `is_valid()`)
        """
        payload = _build_validation_payload(model=self, recursive=recursive)
        
        # Вимикаємо вимагання аліасів, адже ми вже їх промапили в `.elastic_create`
        return self.__class__.model_validate(payload, by_alias=False, by_name=True)

    def elastic_get_model_fields(self) -> Mapping[str, FieldInfo]:
        """
        (ChatGPT advises not to call this in system methods, possible crash accessing nested model fields, but this seems to be false)
        (ЧатГПТ радить не викликати це в системних методах, можливий збій доступа до полів вкладених моделей, але схоже це брехня)
        """
        cls = object.__getattribute__(self, '__class__')
        return cls.model_fields

    # ---------------------------
    # Access/Assignment Behavior
    # ---------------------------
    
    # def __getattribute__(self, name: str) -> Any:
    #     # Quick exits for service attributes and dunders
    #     if name.startswith('_') or name in _SYSTEM_ATTRS:
    #         return object.__getattribute__(self, name)

    #     # Call NotLoadedFieldError if key is not loaded
    #     self._raise_if_not_loaded(model=self, name=name)

    #     # Normal access
    #     return object.__getattribute__(self, name)

    # def __getattr__(self, name: str) -> Any:
    #     """
    #     fallback, which is called if __getattribute__ raised AttributeError/field is not in __dict__.
    #     fallback, який викликається, якщо __getattribute__ підняв AttributeError/поля немає у __dict__.
    #     """
        
    #     # Call NotLoadedFieldError if key is not loaded
    #     self._raise_if_not_loaded(self, name)

    #     raise AttributeError(name)


    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)

        # Актуалізовуємо `elastic_loaded_fields`, видаляємо юзерське поле
        elastic_finished = object.__getattribute__(self, "_elastic_finished")
        if elastic_finished:
            elastic_loaded_fields: set = object.__getattribute__(self, "_elastic_loaded_fields")
            if name in elastic_loaded_fields:
                elastic_loaded_fields.discard(name)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

        # Актуалізовуємо `elastic_loaded_fields`, вписуємо нове юзерське поле
        elastic_finished = object.__getattribute__(self, "_elastic_finished")
        if elastic_finished:
            elastic_loaded_fields = object.__getattribute__(self, "_elastic_loaded_fields")
            elastic_loaded_fields.add(name)
        
    
    # ---------------------------
    # Internal validation/coercion
    # ---------------------------

    @classmethod
    def _elastic_prepare_value(cls, value: Any, annotation: Any, validate: bool) -> Any:
        """
        Головна задача цієї функції це:
        - `validate==True`: підготувати `value` до правильного коерсингу/валідації `pydantic`
            - Шукаємо вкладені `ElasticModel` та конструюємо їх одразу, по логіці `.elastic_create`
        - `validate==False`: отримати оброблений `value` який буде юзатися якщо при вимкнутій валідації, pydantic не буде обробляти цей `value` тому вся обробка на нас
            - Шукаємо `ElasticModel` та конструюємо їх одразу, по логіці `.elastic_create`
            - Шукаємо `BaseModel` та конструюємо їх одразу, інакше в нас буде dict
        """
        base = _strip_annot(annotation)     # EN: remove outer Annotated[...] wrapper; UA: знімаємо зовнішній Annotated[...]
        origin = get_origin(base)

        if value is None:
            return None

        # --- Direct ElasticModel from dict ---
        # EN: If field type is a subclass of ElasticModel and we got a dict, create a partial nested model.
        # UA: Якщо тип поля — підклас ElasticModel і прийшов dict, створюємо часткову вкладену модель.
        
        # !todo на кожному value викликається ця перевірка, можливо потрібно це робити якось оптимізованіше, перевірити що value це клас хочаб
        if isinstance(value, dict):
            
            if _issubclass_safe(base, ElasticModel):
                return base.elastic_create(value, validate=validate)
            
            elif _issubclass_safe(base, BaseModel):
                if validate:
                    return value  # Коеристь та валідувати буде pydantic
                else:
                    # Формуємо BaseModel модель через .model_construct, з підтримкою вкладеностей
                    fields = base.model_fields
                    alias_to_name = { (f.alias or n): n for n, f in fields.items() }
                    raw_ann = _raw_annotations_map(base)
                    basemodel_data = {}
                    
                    for raw_key, v in value.items():
                        name = alias_to_name.get(raw_key, raw_key)
                        if name in fields:
                            ann = raw_ann.get(name, fields[name].annotation)
                            basemodel_data[name] = cls._elastic_prepare_value(value=v, annotation=ann, validate=validate)
                        else:
                            # зберігаємо всі невідомі поля, хай .model_construct сам розбирається що з ними робити
                            basemodel_data[name] = v
                    
                    basemodel = base.model_construct(**basemodel_data)
                    return basemodel


        # --- Union / Optional ---
        if origin is Union:
            # Не робимо перетворень для Union (у т.ч. Optional, BaseModel, ElasticModel). Pydantic сам обере гілку (особливо важливо для дискримінованих Union).
            # !todo Union[BaseModel, BaseModel] (discriminator) не будуть мапиться якщо validate==False, замість них ми отримаємо dict
            return value

        # --- list / set / tuple containers ---
        if origin in (list, set, tuple):
            args = get_args(base) or (Any,)

            # EN: Variadic/Fixed tuple handling – recurse only into annotated positions.
            # UA: Обробка кортежів (варіативних/фіксованих) – рекурсія лише в позиції, описані в анотації.
            if origin is tuple:
                # Variadic: Tuple[T, ...]
                if len(args) == 2 and args[1] is Ellipsis:
                    item_ann = args[0]
                    # EN: map each item through pre-pass; UA: проганяємо кожен елемент через pre-pass
                    return tuple(cls._elastic_prepare_value(x, item_ann, validate) for x in value)
                # Fixed-length: Tuple[T1, T2, ...]
                else:
                    head_count = len(args)
                    head = [
                        cls._elastic_prepare_value(x, a, validate)
                        for x, a in zip(value[:head_count], args)
                    ]
                    tail = list(value[head_count:])  # EN: keep tail as-is; UA: хвіст лишаємо як є
                    return tuple(head + tail)

            # EN: list/set – recurse into items using their item annotation (if present).
            # UA: list/set – рекурсія по елементах за їх анотацією (якщо задана).
            item_ann = args[0]
            if origin is list:
                return [cls._elastic_prepare_value(value=x, annotation=item_ann, validate=validate) for x in value]
            else:  # set
                return {cls._elastic_prepare_value(value=x, annotation=item_ann, validate=validate) for x in value}

        # --- dict[K, V] container ---
        if origin is dict:
            k_ann, v_ann = (get_args(base) or (Any, Any))
            # EN: Only values may contain ElasticModel; keys stay untouched here.
            # UA: Лише значення можуть містити ElasticModel; ключі не чіпаємо тут.
            return {k: cls._elastic_prepare_value(value=v, annotation=v_ann, validate=validate) for k, v in value.items()}

        if base in (int, str):
            pass

        # No prep needed; return as-is for Pydantic to handle.
        return value


    @classmethod
    def _elastic_coerce_value(cls, annotation: Any, value: Any, validate: bool, strict_validate: bool = False) -> Any:
        """
        - Якщо `validate == True`: Приведення `value` до його задекларованого типу та валідація через pydantic, 
        - Якщо `validate == False`: Лише формуємо вкладені `ElasticModel` та `BaseModel` моделі 
        """

        prepared_value = cls._elastic_prepare_value(value, annotation, validate)

        if validate or strict_validate:
            # Pydantic коерсинг та валідація
            adapter = _adapter(annotation)
            return adapter.validate_python(prepared_value, strict=strict_validate)
        else:
            # Повертаємо мінімально оброблену версію
            return prepared_value
