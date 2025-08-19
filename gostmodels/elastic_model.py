"""
ElasticModel — базовий клас для роботи з частковими (projection) документами з БД. 

Він дозволяє створювати об’єкти з урізаних dict, ігноруючи відсутні required‑поля, але:
    - кидатиме помилку при доступі до незавантаженого поля (NotLoadedFieldError);
    - валідовує наявні значення типово через TypeAdapter (email, datetime, enum, …);
    - рекурсивно будує вкладені моделі (які теж наслідують ElasticModel);
    - складає зайві ключі у .extra (без валідації);
    - підтримує два режими перевірки: глибинний і поверхневий.

Типові кейси: 
    - mongo projections
    - часткові відповіді API
    - денормалізовані субдокументи

Коли використовувати:
    - Коли БД/API повертає частину полів (проєкція) і ти не хочеш робити все Optional.
    - Коли треба швидко працювати з неповними документами, але «спіткнутися» при доступі до поля, якого не було в даних.
    - Коли треба відкласти повну валідацію до «моменту істини» (запис у БД, виклик зовнішнього сервісу).

Коли не використовувати:
    - Якщо дані завжди повні і ти бажаєш запускати валідатори одразу — достатньо звичайного model_validate.
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
#   Допоміжні сутності
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
    Код звертається до поля, яке описане в моделі, але не було завантажено.
    Повідомлення підказує: додай це поле в модель, перед звернення до нього.
    """


def _issubclass_safe(tp: Any, base: type) -> bool:
    """
    Безпечна перевірка issubclass для випадків, коли tp може бути не-класом (наприклад, typing-конструкції).
    Використовується лише як безпечна перевірка. Функція гарантує False замість помилки.
    """
    try:
        return isinstance(tp, type) and issubclass(tp, base)
    except TypeError:
        return False


@lru_cache(maxsize=256)
def _raw_annotations_map(cls: type) -> dict[str, Any]:
    # повертає анотації з Annotated/Field(...) всередині
    return get_type_hints(cls, include_extras=True)


@lru_cache(maxsize=512)
def _adapter_hashable(annotation: Any) -> TypeAdapter:
    return TypeAdapter(annotation)

def _adapter(annotation: Any) -> TypeAdapter:
    """
    Кешований фабричний метод для TypeAdapter(annotation).
    TypeAdapter в Pydantic v2 — це валідатор/коерсер за type-hint'ом без створення BaseModel.
    Створення адаптера не безкоштовне, тому кеш суттєво зменшує накладні витрати.

    - Мета: 
        - Отримати кешований pydantic.TypeAdapter(annotation) — механізм, який валідовує/коерсить значення за type-hint’ом без створення моделі.
    - Чому кеш:
        - Значно пришвидшує повторні валідації однакових типів,
    - Нотатки:
        - Ключем кешу є сам annotation. Якщо ти передаєш той самий об’єкт типу (наприклад, EmailStr, list[int], MyModel), адаптер береться з кешу.
    """
    # Annotated[..., FieldInfo] може виявитися негешабельним (бо всередині є FieldInfo).
    try:
        return _adapter_hashable(annotation)  # спроба кеша за хешем
    except TypeError:
        # напр. Annotated[..., Field(...)] часто не хешується → це очікувано
        logger.warning(
            "ElasticModel: unhashable annotation for adapter cache; using non-cached adapter: %r",
            annotation,
        )
        return TypeAdapter(annotation)      # fallback без кеша
    except Exception:
        logger.exception(
            "ElasticModel: unexpected error creating TypeAdapter for %r; falling back to non-cached instance",
            annotation,
        )
        return TypeAdapter(annotation)        # fallback без кеша


def _strip_annot(annotation: Any) -> Any:
    """
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
    Формує payload для валідації через BaseModel.model_validate(...).
    
    Параметри:
    - recursive:
        - Якщо True - Повна серіалізація: вся модель і вкладені BaseModel перетворюються на dict/list/...
        - Якщо False - Поверхнева серіалізація. Вкладені моделі залишаються інстансами (не перетворюються в dict)

    Навіщо потрібна поверхнева серіалізація (`recursive == False`):
        - Якщо викликати BaseModel.model_validate і передавати dict - буде виконана повна валідація з формуванням вкладених моделей
        - Якщо викликати BaseModel.model_validate і передавати dict який має в собі інстанси вкладених моделей - ці моделі не будуть повторно створюватись і валіуватись, а залишаться як є 
            - Увага. Лише за умови ConfigDict.revalidate_instances == 'never' (default)

    Зауваги:
        - Якщо `recursive=False` у полі випадково лежить dict (а не інстанс BaseModel), Pydantic обробить його як сирі дані та піде в глибину для цього поля.
        - Якщо `ConfigDict.revalidate_instances != 'never'`, то навіть інстанси вкладених моделей будуть перевалідовані.
    """
    # Повна серіалізація
    if recursive:
        data = model.model_dump(exclude_unset=True)
        return data
    # Поверхнева серіалізація. 
    # Беремо лише реально завантажені поля (або вручну присвоєні через __setattr__)
    else:
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
#   Основний клас
# =========================

class ElasticModel(BaseModel):
    """
    Клас обгортка 'BaseModel' яка дозволяє конструювати об'єкти моделі з різним набором даних:
        - Без необхідності мати всі поля
        - З зайвими для моделі, але важливими для контексту полями, які будуть збережені в `.extra`, для доступу до них

    Це дозволить нам створювати моделі з обмеженими/надлишковими даними, які отримані з зовншніх ресурсів (наприклад БД)
        
    Механіка:
    - Створюй екземпляри з projection-документів через elastic_create().
    - Доступ до незавантажених полів кидає NotLoadedFieldError.
    - Зайві ключі доступні у .extra (без валідації).
    - Вкладені ElasticModel працюють рекурсивно з тією ж семантикою.
    - Для "повної" валідації та запуску валідаторів класу використовуй to_validated().

    """

    model_config = ConfigDict(
        extra='ignore',                 # Ігноруємо лишні ключі на рівні моделі, але зберігаємо їх в ._extra для ручного доступу.
        populate_by_name=True,          # Дозволяє підставляти дані і за alias, і за ім'ям поля
        revalidate_instances='never'    # Не виконуємо валідацію вкладеного об'єкта, якщо він вже є інстансом BaseModel
    )

    # Приватні поля-носії службового стану
    _extra: dict[str, Any] = PrivateAttr(default_factory=dict)  # Усі невідомі моедлі поля зберігаються тут (без валідації)
    _loaded_fields: set[str] = PrivateAttr(default_factory=set)


    @classmethod
    def elastic_create(
        cls,
        data: dict[str, Any],
        *,
        validate: bool = True,
        apply_defaults: bool = False,
    ) -> Self:
        """
        Побудувати partial-екземпляр класу з урізаного dict.


        :param data: `dict` (може містити частину полів та "зайві" ключі які попадуть у `.extra`).
        :param validate: 
            - Якщо True (за замовчуванням) - усі значення з `data` будуть провалідовані та перетворені за їх тип-анотаціями в моделі.
            - Якщо False - значення приймаються "як є" (лише для довірених потоків).
        :param apply_defaults:
            - Якщо True - для відустніх полів підставляються `default`/`default_factory`.
            - Якщо False - відсутні поля не створюються; звертання до них кине `NotLoadedFieldError`.

        :return Eкземпляр класу з:
            - Встановленими лише тими полями, які знаходяться у `data` (Та дефолтними значеннями, якщо `apply_defaults == True`)

            - `.extra` - слованик з усіма невідомими ключами на рівні моделі (без валідації), 
            (Якщо полів передано більше чим описано в моделі, вони будуть знаходитись в цьому словнику);

            - `._loaded_fields` - список (множина) полів, які реально були встановлені.

        Увага:
        - Валідатори класу (`field_validator`/`model_validator`) на цьому етапі НЕ запускаються. Запускай їх через `to_validated()` або звичайний `model_validate()`.
        """
        fields = cls.model_fields
        alias_to_name = {f.alias or n: n for n, f in fields.items()}    # Підтримка alias: наприклад, "_id" -> "id"
        raw_ann = _raw_annotations_map(cls)
        
        provided: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        # Розкладаємо вхідні дані на відомі/зайві; коерсінг/валідація наявних значень.
        for raw_key, value in data.items():
            name = alias_to_name.get(raw_key, raw_key)
            
            if name in fields:
                annotation = raw_ann.get(name, fields[name].annotation)
                provided[name] = cls._coerce(annotation, value, validate)
            else:
                extra[raw_key] = value  # "зайве" — без валідації

        # Підставляємо дефолти для відсутніх полів (Опційно)
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

        # Створюємо інстанс, не вимагаючи повноти (partial-конструктор).
        inst = cls.model_construct(**provided)

        # Зберігаємо службову інформацію.
        object.__setattr__(inst, "_extra", extra)
        object.__setattr__(inst, "_loaded_fields", set(provided.keys()))
        return inst

    @property
    def extra(self) -> dict[str, Any]:
        """
        Усі невідомі моделі поля зберігаються тут (без валідації),
        """
        return self._extra

    def is_loaded(self, name: str) -> bool:
        """
        Перевіряє, Чи було поле встановлене під час `elastic_create``.
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
    
    def is_valid(self, *, recursive: bool = True) -> tuple[bool, list[str]]:
        """
        Перевіряє валідність без повернення нового екземпляра.
        Повертає (True/False, bad_paths: List[str]), де bad_paths — 'a.b[2].c' тощо.
        """
        payload = _build_validation_payload(model=self, recursive=recursive)
        
        try:
            _adapter(self.__class__).validate_python(payload)
            return True, []
        except ValidationError as e:
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

    def get_validated_model(self, recursive: bool = True) -> Self:
        """
        Повна валідація поточного стану моделі:
        - Якщо валідація успішна - повертає новий екземпляр класу
        - Якшо валідація не успішна - кидає ValidationError.

        (Якщо хочеш лише дізнатись, чи валідний цей об'єкт — користуйся `is_valid()`)
        """
        payload = _build_validation_payload(model=self, recursive=recursive)
        return self.__class__.model_validate(payload)


    def get_model_fields(self) -> Mapping[str, FieldInfo]:
        """
        ( ЧатГПТ радить не викликати це в системних методах, можливий збій доступа до полів вкладених моделей, але схоже це брехня)
        """
        cls = object.__getattribute__(self, '__class__')
        return cls.model_fields

    # ---------------------------
    # Поведінка доступу/присвоєння
    # ---------------------------
    
    def __getattribute__(self, name: str) -> Any:
        # Швидкі виходи для службових атрибутів і dunder'ів
        if name.startswith('_') or name in _SYSTEM_ATTRS:
            return object.__getattribute__(self, name)

        # Викликаємо NotLoadedFieldError, якщо key не завантажено
        _raise_if_not_loaded(self, name)

        # Звичайний доступ
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        """
        fallback, який викликається, якщо __getattribute__ підняв AttributeError/поля немає у __dict__.
        """
        
        # Викликаємо NotLoadedFieldError, якщо key не завантажено
        _raise_if_not_loaded(self, name)

        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Ручне присвоєння поля моделі також відмічає його як «завантажене».
        Не зачіпає службові/приватні атрибути.
        """
        super().__setattr__(name, value)
        try:
            model_fields = type(self).model_fields
            if name in model_fields:
                # якщо _loaded_fields ще не ініціалізований (під час конструкції) — пропустимо
                lf = object.__getattribute__(self, "_loaded_fields")
                lf.add(name)
        except AttributeError:
            logger.warning(
                "ElasticModel.__setattr__: '_loaded_fields' not ready on %s while setting '%s'",
                type(self).__name__, name,
            )
            pass  # під час ранньої ініціалізації приватних атрибутів
    
    # ---------------------------
    # Внутрішня валідація/коерсинг
    # ---------------------------

    @classmethod
    def _coerce(cls, annotation: Any, value: Any, validate: bool) -> Any:
        """
        Коерсинг/валідація `value` значення згідно `annotation`.
        """
        raw = annotation                     # оригінальна анотація (з метаданими/constraints)
        base = _strip_annot(annotation)      # базовий тип для аналізу структури
        origin = get_origin(base)

        # None
        if value is None:
            return _adapter(raw).validate_python(None) if validate else None
        
        # Вкладена ElasticModel з dict
        if _issubclass_safe(base, ElasticModel) and isinstance(value, dict):
            return base.elastic_create(value, validate=validate)

        # Union / Optional — делегуємо повністю
        if origin is Union:
            return _adapter(raw).validate_python(value) if validate else value

        # Контейнери (list/set/tuple): обходимо елементи рекурсивно
        if origin in (list, set, tuple):
            args_base = get_args(base)  # структурні аргументи (можуть містити Annotated всередині)
            
            # Tuple. Приводимо до позиційних типів, перевіряємо фіксовану довжину
            if origin is tuple:
                # Tuple[T, ...] — варіативний
                if len(args_base) == 2 and args_base[1] is Ellipsis:
                    item_ann = args_base[0]
                    out_tuple = tuple(cls._coerce(item_ann, x, validate) for x in value)

                    if validate:    # проганяємо готовий tuple через повну валідацію за "raw"
                        return _adapter(raw).validate_python(out_tuple)
                    else:
                        return out_tuple
                
                # Tuple[T1, T2, ...] — фіксована довжина
                else:
                    spec = list(args_base)  # позиційні raw-анотації (з Annotated/constraints)
                    expected = len(spec)

                    # віддаємо на повну валідацію, щоб отримати коректний ValidationError
                    if validate and len(value) != expected:
                        return _adapter(raw).validate_python(value)
                    
                    # коерсинг голови (за позиційними annotation’ами)
                    head = [cls._coerce(t_ann, x, validate) for t_ann, x in zip(spec, value)]
                    # хвіст лишаємо як є (без коерсингу)
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

        # Контейнер: Dict[K, V]: ключі (за потреби) валідую TypeAdapter’ом, значення — рекурсивно
        if origin is dict:
            args_base = get_args(base)
            key_ann, val_ann = (args_base if args_base else (Any, Any))
            if validate:
                return {_adapter(key_ann).validate_python(k): cls._coerce(val_ann, v, validate) for k, v in value.items()}
            else:
                # без перевірки ключів у режимі validate=False
                return {k: cls._coerce(val_ann, v, validate) for k, v in value.items()}

        # Усе інше (int/str/EmailStr/Decimal/datetime/Enum/AnyUrl/...) — делегуємо TypeAdapter
        return _adapter(raw).validate_python(value) if validate else value


def _raise_if_not_loaded(model: "ElasticModel", name: str) -> None:
    """
    Єдине місце правди: якщо `name` є полем моделі, але не в `_loaded_fields` - піднімаємо NotLoadedFieldError.
    """
    cls = object.__getattribute__(model, '__class__')
    model_fields = cls.model_fields
    if name not in model_fields:
        return
    
    try:
        loaded_fields = object.__getattribute__(model, '_loaded_fields')
    except AttributeError:
        # рання фаза ініціалізації — просто пропускаємо перевірку (тест `test_discriminated_union_validate_true` - не зміг звернутися до вкладених полів вкладених моделей)
        logger.debug(
            "ElasticModel: access to not-loaded field '%s' on model '%s'",
            name, cls.__name__,
        )
        return
    
    if name in loaded_fields:
        return

    raise NotLoadedFieldError(f"Field '{name}' of model '{cls.__name__}' was not loaded.")
