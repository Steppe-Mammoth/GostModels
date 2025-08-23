"""
ElasticModel is a base class for working with partial (projection) documents from a database.

It allows you to create objects from truncated dicts, ignoring missing required fields, but:
- will throw an error when accessing an unloaded field (NotLoadedFieldError);
- validates existing values typically via TypeAdapter (email, datetime, enum, …);
- recursively builds nested models (which also inherit ElasticModel);
- puts extra keys in .extra (without validation);
- supports two validation modes: deep and shallow.

ElasticModel — базовий клас для роботи з частковими (проекційними) документами з бази даних.

Дозволяє створювати об'єкти з урізаних dict'ів, ігноруючи відсутні обов'язкові поля, але:
- кидає помилку при доступі до незавантаженого поля (NotLoadedFieldError);
- валідує наявні значення зазвичай через TypeAdapter (email, datetime, enum, …);
- рекурсивно будує вкладені моделі (які також наслідують ElasticModel);
- поміщає зайві ключі в .extra (без валідації);
- підтримує два режими валідації: глибокий та поверхневий.
"""

from __future__ import annotations


from typing import Any, Annotated, Mapping, Self, Union, get_args, get_origin, get_type_hints
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, PrivateAttr, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

import logging
logger = logging.getLogger(__name__)

# =========================
#   Helper Entities 
# =========================

_SYSTEM_ATTRS = (
    'model_config',
    '__dict__',
    '__class__',
    '__fields_set__',
    '__pydantic_fields_set__',
    'is_loaded',
    'get_model_fields'
)


class NotLoadedFieldError(AttributeError):
    """
    The code accesses a field that is described in the model, but has not been loaded.
    The message prompts: add this field to the model before accessing it.
    
    Код звертається до поля, яке описано в моделі, але не було завантажене.
    Повідомлення підказує: додайте це поле до моделі перед доступом до нього.
    """


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


def _build_validation_payload(model: ElasticModel, recursive: bool) -> dict[str, Any]:
    """
    Forms payload for validation via BaseModel.model_validate(...).
    Формує payload для валідації через BaseModel.model_validate(...).
    
    Parameters / Параметри:
    - recursive:
        - If True - Full serialization: entire model and nested BaseModels are converted to dict/list/...
        - If False - Shallow serialization. Nested models remain instances (not converted to dict)
        - Якщо True - Повна серіалізація: вся модель і вкладені BaseModel перетворюються на dict/list/...
        - Якщо False - Поверхнева серіалізація. Вкладені моделі залишаються інстансами (не перетворюються в dict)

    Why shallow serialization is needed (`recursive == False`):
        - If you call BaseModel.model_validate and pass dict - full validation will be performed with nested model formation
        - If you call BaseModel.model_validate and pass dict which contains nested model instances - these models won't be recreated and validated, but will remain as is
            - Note. Only if ConfigDict.revalidate_instances == 'never' (default)
        
        - Якщо викликати BaseModel.model_validate і передавати dict - буде виконана повна валідація з формуванням вкладених моделей
        - Якщо викликати BaseModel.model_validate і передавати dict який має в собі інстанси вкладених моделей - ці моделі не будуть повторно створюватись і валіуватись, а залишаться як є 
            - Увага. Лише за умови ConfigDict.revalidate_instances == 'never' (default)

    Notes / Зауваги:
        - If `recursive=False` and a dict (not BaseModel instance) accidentally lies in a field, Pydantic will process it as raw data and go deep for this field.
        - If `ConfigDict.revalidate_instances != 'never'`, then even nested model instances will be revalidated.
        - Якщо `recursive=False` у полі випадково лежить dict (а не інстанс BaseModel), Pydantic обробить його як сирі дані та піде в глибину для цього поля.
        - Якщо `ConfigDict.revalidate_instances != 'never'`, то навіть інстанси вкладених моделей будуть перевалідовані.
    """

    if recursive:
        # Full serialization 
        data = model.model_dump(exclude_unset=True)
        return data
    else:
        # Shallow serialization 
        # Take only actually loaded fields (or manually assigned via __setattr__)
        # Беремо лише реально завантажені поля (або вручну присвоєні через __setattr__)
        try:
            loaded_fields = object.__getattribute__(model, '_loaded_fields')
        except AttributeError:
            logger.warning(
                "ElasticModel: '_loaded_fields' is missing on %s; assuming empty set for non-recursive payload",
                type(model).__name__,
            )
            loaded_fields = set()

        model_data = object.__getattribute__(model, '__dict__')
        data = {name: model_data[name] for name in loaded_fields if name in model_data}
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
        extra='ignore',                 # Ignore extra keys at model level, but save them in ._extra for manual access. / Ігноруємо лишні ключі на рівні моделі, але зберігаємо їх в ._extra для ручного доступу.
        populate_by_name=True,          # Allows substituting data by both alias and field name
        revalidate_instances='never'    # Don't validate nested object if it's already a BaseModel instance / Не валідуємо вкладений об'єкт, якщо він вже є інстансом BaseModel

    )

    # Private state-carrying fields
    _extra: dict[str, Any] = PrivateAttr(default_factory=dict)  # All unknown model fields are stored here (without validation)
    _loaded_fields: set[str] = PrivateAttr(default_factory=set) # All loaded model fields are stored here


    @classmethod
    def elastic_create(
        cls,
        data: dict[str, Any],
        *,
        validate: bool = True,
        apply_defaults: bool = False,
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

            - `.extra` - dictionary with all unknown keys at model level (without validation), 
            (If more fields are passed than described in the model, they will be in this dictionary);

            - `._loaded_fields` - list (set) of fields that were actually set.

        Note:
        - Class validators (`field_validator`/`model_validator`) are NOT run at this stage. Run them via `to_validated()` or regular `model_validate()`.
        """
        fields = cls.model_fields
        alias_to_name = {f.alias or n: n for n, f in fields.items()}    # Support alias: e.g., "_id" -> "id"
        raw_ann = _raw_annotations_map(cls)
        
        provided: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        # Decompose input data into known/extra; coercion/validation of existing values.
        for raw_key, value in data.items():
            name = alias_to_name.get(raw_key, raw_key)
            
            if name in fields:
                annotation = raw_ann.get(name, fields[name].annotation)
                provided[name] = cls._coerce(annotation, value, validate)
            else:
                extra[raw_key] = value  # "extra" — without validation

        # Substitute defaults for missing fields (Optional)
        if apply_defaults:
            for name, f in fields.items():
                if name in provided:
                    continue
                has_default = getattr(f, "default", PydanticUndefined) is not PydanticUndefined
                has_factory = getattr(f, "default_factory", None) is not None
                if has_default:
                    provided[name] = getattr(f, "default")
                elif has_factory:
                    provided[name] = f.default_factory()  # type: ignore[attr-defined]

        # Create instance without requiring completeness (partial constructor).
        inst = cls.model_construct(**provided)

        # Save service information.
        object.__setattr__(inst, "_extra", extra)
        object.__setattr__(inst, "_loaded_fields", set(provided.keys()))
        return inst

    @property
    def elastic_extra(self) -> dict[str, Any]:
        """
        All unknown model fields are stored here (without validation).
        """
        return self._extra

    def elastic_is_loaded(self, name: str) -> bool:
        """
        Checks whether the field was set during `elastic_create`.
        """
        try:
            loaded = object.__getattribute__(self, "_loaded_fields")
        except AttributeError:
            logger.warning(
                "ElasticModel: '_loaded_fields' not initialized yet on %s while checking is_loaded('%s')",
                type(self).__name__, name,
            )
            return False
        
        return name in loaded
    
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
        return self.__class__.model_validate(payload)

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
    
    def __getattribute__(self, name: str) -> Any:
        # Quick exits for service attributes and dunders
        if name.startswith('_') or name in _SYSTEM_ATTRS:
            return object.__getattribute__(self, name)

        # Call NotLoadedFieldError if key is not loaded
        self._raise_if_not_loaded(model=self, name=name)

        # Normal access
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        """
        fallback, which is called if __getattribute__ raised AttributeError/field is not in __dict__.
        fallback, який викликається, якщо __getattribute__ підняв AttributeError/поля немає у __dict__.
        """
        
        # Call NotLoadedFieldError if key is not loaded
        self._raise_if_not_loaded(self, name)

        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Manual assignment of model field also marks it as "loaded".
        Doesn't touch service/private attributes.
        """
        super().__setattr__(name, value)
        try:
            model_fields = type(self).model_fields
            if name in model_fields:
                # if _loaded_fields is not yet initialized (during construction) — skip
                # якщо _loaded_fields ще не ініціалізований (під час конструкції) — пропустимо
                lf = object.__getattribute__(self, "_loaded_fields")
                lf.add(name)
        except AttributeError:
            logger.warning(
                "ElasticModel.__setattr__: '_loaded_fields' not ready on %s while setting '%s'",
                type(self).__name__, name,
            )
            pass  # during early initialization of private attributes / під час ранньої ініціалізації приватних атрибутів
    
    # ---------------------------
    # Internal validation/coercion
    # ---------------------------

    @classmethod
    def _coerce(cls, annotation: Any, value: Any, validate: bool) -> Any:
        """
        Coercion/validation of `value` according to `annotation`.
        """
        raw = annotation                     # original annotation (with metadata/constraints) / оригінальна анотація (з метаданими/constraints)
        base = _strip_annot(annotation)      # base type for structure analysis / базовий тип для аналізу структури
        origin = get_origin(base)

        # None
        if value is None:
            return _adapter(raw).validate_python(None) if validate else None
        
        # Nested ElasticModel from dict
        if _issubclass_safe(base, ElasticModel) and isinstance(value, dict):
            return base.elastic_create(value, validate=validate)

        # Union / Optional — delegate completely
        if origin is Union:
            return _adapter(raw).validate_python(value) if validate else value

        # Containers (list/set/tuple): traverse elements recursively
        if origin in (list, set, tuple):
            args_base = get_args(base)  # structural arguments (may contain Annotated inside) / структурні аргументи (можуть містити Annotated всередині)
            
            # Tuple. Convert to positional types, check fixed length / Приводимо до позиційних типів, перевіряємо фіксовану довжину
            if origin is tuple:
                # Tuple[T, ...] — variadic
                if len(args_base) == 2 and args_base[1] is Ellipsis:
                    item_ann = args_base[0]
                    out_tuple = tuple(cls._coerce(item_ann, x, validate) for x in value)

                    if validate:    # run ready tuple through full validation by "raw"
                        return _adapter(raw).validate_python(out_tuple)
                    else:
                        return out_tuple
                
                # Tuple[T1, T2, ...] — fixed length
                else:
                    spec = list(args_base)  # positional raw-annotations (with Annotated/constraints) / позиційні raw-анотації (з Annotated/constraints)
                    expected = len(spec)

                    # give to full validation to get correct ValidationError / віддаємо на повну валідацію, щоб отримати коректний ValidationError
                    if validate and len(value) != expected:
                        return _adapter(raw).validate_python(value)
                    
                    # coerce head (by positional annotations) / коерсинг голови (за позиційними annotation'ами)
                    head = [cls._coerce(t_ann, x, validate) for t_ann, x in zip(spec, value)]
                    # leave tail as is (without coercion) / хвіст лишаємо як є (без коерсингу)
                    tail = list(value[expected:]) if len(value) > expected else []
                    
                    out_tuple = tuple(head + tail)
                    if validate:
                        return _adapter(raw).validate_python(out_tuple)
                    else:
                        return out_tuple

            # List/Set
            item_ann = args_base[0] if args_base else Any
            items = (cls._coerce(item_ann, x, validate) for x in value)
            if origin is list:
                out_array = list(items)
            else:
                out_array = set(items)
            
            if validate:
                return _adapter(raw).validate_python(out_array)
            else:
                return out_array

        # Container: Dict[K, V]: keys (if needed) validated by TypeAdapter, values — recursively
        if origin is dict:
            args_base = get_args(base)
            key_ann, val_ann = (args_base if args_base else (Any, Any))
            if validate:
                return {_adapter(key_ann).validate_python(k): cls._coerce(val_ann, v, validate) for k, v in value.items()}
            else:
                # without key checking in validate=False mode
                return {k: cls._coerce(val_ann, v, validate) for k, v in value.items()}

        # Everything else (int/str/EmailStr/Decimal/datetime/Enum/AnyUrl/...) — delegate to TypeAdapter
        return _adapter(raw).validate_python(value) if validate else value

    @classmethod
    def _raise_if_not_loaded(cls, model: "ElasticModel", name: str) -> None:
        """
        Single source of truth: if `name` is a model field but not in `_loaded_fields` - raise NotLoadedFieldError.
        Єдине місце правди: якщо `name` є полем моделі, але не в `_loaded_fields` - піднімаємо NotLoadedFieldError.
        """
        cls = object.__getattribute__(model, '__class__')
        model_fields = cls.model_fields
        if name not in model_fields:
            return
        
        try:
            loaded_fields = object.__getattribute__(model, '_loaded_fields')
        except AttributeError:
            # early initialization phase — just skip the check (test `test_discriminated_union_validate_true` - couldn't access nested fields of nested models)
            # рання фаза ініціалізації — просто пропускаємо перевірку (тест `test_discriminated_union_validate_true` - не зміг звернутися до вкладених полів вкладених моделей)
            logger.debug(
                "ElasticModel: access to not-loaded field '%s' on model '%s'",
                name, cls.__name__,
            )
            return
        
        if name in loaded_fields:
            return

        raise NotLoadedFieldError(f"Field '{name}' of model '{cls.__name__}' was not loaded.")
