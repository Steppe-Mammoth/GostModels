# Покривають:
# - partial-створення з проєкцій
# - строгий доступ до незавантажених полів (NotLoadedFieldError)
# - рекурсію для вкладених моделей
# - extra-поля
# - alias
# - shallow vs deep валідацію (is_valid / get_validated_model)
# - колекції (list/set/tuple), dict-ключі, Union (дискримінований)
# - Annotated-constraints
# - оновлення _loaded_fields при ручному присвоєнні (__setattr__)


import pytest
from typing import Annotated, Literal, Union

from pydantic import EmailStr, Field, ValidationError

from gostmodels import ElasticModel, NotLoadedFieldError

# --------------------------
# Моделі для тестів
# --------------------------

class Created(ElasticModel):
    at: str
    by: str


class Password(ElasticModel):
    hash: str
    salt: str


class User(ElasticModel):
    id: str = Field(alias="_id")
    first_name: str
    last_name: str
    middle_name: str | None = None
    email: EmailStr
    phone: str
    created: Created
    password: Password
    # Дефолтне поле: важливо перевірити строгий доступ (має кидати, якщо не завантажене)
    flag: bool = False

    # Для списків/масивів користувачів (для перевірки шляхів типу members[0].email)
class UserSlim(ElasticModel):
    email: EmailStr
    created: Created


class Group(ElasticModel):
    name: str
    members: list[UserSlim]


# Для диктів з типізованими ключами/значеннями
class KV(ElasticModel):
    data: dict[int, float]


# Для колекцій і кортежів
class Containers(ElasticModel):
    nums_list: list[int]
    nums_set: set[int]
    pair_fixed: tuple[int, str]
    any_tail: tuple[int, ...]  # варіативний Tuple[T, ...]


# Для Annotated-constraints
class WithAnnotated(ElasticModel):
    short: Annotated[str, Field(min_length=3)]  # мін. довжина — 3


# Для Union (дискримінований)
class Card(ElasticModel):
    kind: Literal["card"]
    pan: str


class Bank(ElasticModel):
    kind: Literal["bank"]
    iban: str


class Payment(ElasticModel):
    method: Union[Card, Bank] = Field(..., discriminator="kind")
    #method: Card


def test_partial_creation_and_strict_access():
    """
    Часткове створення User з мінімального набору полів.
    Доступ до відсутніх полів → NotLoadedFieldError (строгий доступ).
    """
    doc = {
        "_id": "u1",
        "first_name": "Ann",
        "last_name": "Lee",
        "created": {"at": "2025-08-15"},
        "password": {"hash": "h", "salt": "s"},
        "phone": "+1",
        "email": "ann@example.com",
    }
    u = User.elastic_create(doc, validate=True, apply_defaults=False)

    # Завантажені поля читаються
    assert u.first_name == "Ann"
    assert u.created.at == "2025-08-15"

    # НЕЛОАДЕД вкладене поле → помилка
    with pytest.raises(NotLoadedFieldError):
        _ = u.created.by

    # Дефолтне top-level поле, але без apply_defaults → теж помилка
    with pytest.raises(NotLoadedFieldError):
        _ = u.flag


def test_extra_fields_collected_in_extra():
    """
    Невідомі ключі мають складатися у .extra без валідації.
    """
    doc = {
        "_id": "u1",
        "first_name": "Ann",
        "last_name": "Lee",
        "created": {"at": "2025-08-15", "by": "sys"},
        "password": {"hash": "h", "salt": "s"},
        "phone": "+1",
        "email": "ann@example.com",
        "debug_flag": 1,  # зайве поле
    }
    u = User.elastic_create(doc)
    assert u.extra["debug_flag"] == 1
    # .extra не впливає на схему
    model_fields = u.get_model_fields()
    assert "debug_flag" not in model_fields


def test_alias_support_for_id():
    """
    Перевірка alias: 'id' має ініціалізуватись із '_id'.
    """
    u = User.elastic_create({
        "_id": "x-1",
        "first_name": "A",
        "last_name": "B",
        "email": "a@b.com",
        "phone": "123",
        "created": {"at": "t", "by": "b"},
        "password": {"hash": "h", "salt": "s"},
    })
    assert u.id == "x-1"


def test_apply_defaults_top_level():
    """
    apply_defaults=True → підставити дефолти top-level полів і помітити їх як loaded.
    """
    doc = {
        "_id": "u1",
        "first_name": "Ann",
        "last_name": "Lee",
        "email": "ann@example.com",
        "phone": "+1",
        "created": {"at": "t", "by": "sys"},
        "password": {"hash": "h", "salt": "s"},
    }
    u = User.elastic_create(doc, apply_defaults=True)
    assert u.flag is False
    # is_loaded має сказати, що цей дефолт тепер «завантажений»
    assert u.is_loaded("flag") is True


def test_nested_model_partial_and_strict_access():
    """
    Вкладена ElasticModel має таку ж семантику: partial + строгий доступ.
    """
    u = User.elastic_create({
        "_id": "u1",
        "first_name": "Ann",
        "last_name": "Lee",
        "email": "ann@example.com",
        "phone": "+1",
        "created": {"at": "2025-08-15"},           # без 'by'
        "password": {"hash": "h", "salt": "s"},
    })

    # Доступ до created.by → помилка
    with pytest.raises(NotLoadedFieldError):
        _ = u.created.by


def test_is_valid_shallow_vs_deep_paths():
    """
    is_valid(recursive=True) має показати помилки у вкладених полях ('created.by').
    is_valid(recursive=False) перевіряє тільки верхній рівень і не лізе вглиб.
    """
    u = UserSlim.elastic_create({
        "email": "a@b.com",
        "created": {"at": "t"},  # немає 'by'
    })

    ok_deep, bad_deep = u.is_valid(recursive=True)
    assert ok_deep is False
    # 'created.by' має бути серед шляхів помилок
    assert any(p == "created.by" for p in bad_deep)

    ok_shallow, bad_shallow = u.is_valid(recursive=False)
    # shallow не перевіряє вкладені інстанси → має не падати по created.by
    assert ok_shallow is True
    assert all(p != "created.by" for p in bad_shallow)


def test_get_validated_model_shallow_and_deep():
    """
    get_validated_model(recursive=False) не перевалідовує вкладені інстанси,
    отже пропустить відсутній created.by. А от recursive=True — кине ValidationError.
    """
    u = UserSlim.elastic_create({
        "email": "a@b.com",
        "created": {"at": "t"},  # без 'by'
    })

    # Shallow: працює (вкладений інстанс приймається «як є»)
    u_shallow = u.get_validated_model(recursive=False)
    assert isinstance(u_shallow, UserSlim)

    # Deep: має впасти
    with pytest.raises(ValidationError):
        _ = u.get_validated_model(recursive=True)


def test_dict_key_and_value_coercion_validate_true_and_false():
    """
    dict[int, float]:
    - validate=True → ключі та значення коерсяться (ключ '1' → 1, '2.5' → 2.5)
    - validate=False → лишаються «як є»
    """
    d = {"data": {"1": "2.5", "2": 3}}
    m_true = KV.elastic_create(d, validate=True)
    assert m_true.data == {1: 2.5, 2: 3.0}

    m_false = KV.elastic_create(d, validate=False)
    assert m_false.data == {"1": "2.5", "2": 3}


def test_list_and_set_coercion():
    """
    list[int] і set[int]:
    - validate=True → елементи приводяться до int
    - validate=False → залишаються як є
    """
    c_true = Containers.elastic_create({
        "nums_list": ["1", 2, "3"],
        "nums_set":  {"1", 2, "3"},
        "pair_fixed": [1, "x"],
        "any_tail":  ["4", "5", 6],
    }, validate=True)

    assert c_true.nums_list == [1, 2, 3]
    assert c_true.nums_set == {1, 2, 3}
    assert c_true.pair_fixed == (1, "x")
    assert c_true.any_tail == (4, 5, 6)

    c_false = Containers.elastic_create({
        "nums_list": ["1", 2, "3"],
        "nums_set":  {"1", 2, "3"},
        "pair_fixed": [1, "x"],
        "any_tail":  ["4", "5", 6],
    }, validate=False)

    assert c_false.nums_list == ["1", 2, "3"]
    assert c_false.nums_set == {"1", 2, "3"}  # тип множини зберігається, але елементи не коерсяться
    assert c_false.pair_fixed == (1, "x")
    assert c_false.any_tail == ("4", "5", 6)  # для Tuple[T, ...] при validate=False елементи не коерсяться


def test_tuple_fixed_validate_true_and_false_no_truncation():
    """
    pair_fixed: tuple[int, str]
    - validate=True і невірна довжина → делегуємо у Pydantic (ValidationError)
    - validate=False і довша послідовність → коерсимо тільки відомі позиції, хвіст не обрізаємо
    """
    # validate=True: зайвий елемент → має впасти
    with pytest.raises(ValidationError):
        _ = Containers.elastic_create({
            "nums_list": [],
            "nums_set": set(),
            "pair_fixed": [1, "x", "EXTRA"],  # довжина 3 замість 2
            "any_tail": [],
        }, validate=True)

    # validate=False: не обрізаємо хвіст; перші 2 елементи коерсяться
    c = Containers.elastic_create({
        "nums_list": [],
        "nums_set": set(),
        "pair_fixed": ["1", "x", 99],  # три елементи
        "any_tail": [],
    }, validate=False)

    assert c.pair_fixed == ( "1", "x", 99 )
    # ВАЖЛИВО: згідно з твоєю реалізацією для validate=False
    #   - перші позиції коерсяться за схемою (int -> тут validate=False, тож елемент залишився як є: "1")
    #   - хвіст не коерситься і не обрізається (залишився 99):
    # Якщо ти змінюватимеш політику коерсингу для validate=False (наприклад, "tail-type"),
    # цей тест треба буде оновити.


def test_discriminated_union_validate_true():
    """
    Union із дискримінатором: validate=True → будується коректний підтип.
    """
    p = Payment.elastic_create({
        "method": {"kind": "card", "pan": "1234", "xxx": 1}
    }, validate=True)

    assert isinstance(p.method, Card)
    assert p.method.pan == "1234"
    


def test_discriminated_union_validate_false_keeps_raw():
    """
    Union із дискримінатором: validate=False → лишається «як є» (dict),
    бо для Union гілка у _coerce повертає value без змін.
    """
    p = Payment.elastic_create({
        "method": {"kind": "bank", "iban": "UA..."}
    }, validate=False)

    assert isinstance(p.method, dict)
    assert p.method["kind"] == "bank"
    assert p.method["iban"] == "UA..."


def test_annotated_constraints_preserved():
    """
    Annotated[str, Field(min_length=3)]:
    - validate=True → має врахувати constraint (мін. довжина) і впасти для "ab".
    - validate=False → пропускаємо як є.
    """
    # validate=True → ValidationError
    with pytest.raises(ValidationError):
        result = WithAnnotated.elastic_create({"short": "ac"}, validate=True)


    # validate=False → "ab" приймається як є
    m = WithAnnotated.elastic_create({"short": "ab"}, validate=False)
    assert m.short == "ab"


def test_manual_assignment_marks_loaded_and_allows_access():
    """
    __setattr__ має позначати поле як «завантажене», аби строгий доступ дозволив читання.
    """
    # Створюємо частково, без email
    u = UserSlim.elastic_create({
        "created": {"at": "t", "by": "sys"}
    })

    with pytest.raises(NotLoadedFieldError):
        _ = u.email

    # Ручне присвоєння → має стати «loaded»
    u.email = "x@y.z"
    assert u.is_loaded("email") is True
    assert u.email == "x@y.z"


def test_group_members_paths_in_deep_validation():
    """
    Перевірка форматування шляхів у масивах: members[0].email.
    """
    g = Group.elastic_create({
        "name": "admins",
        "members": [
            {"email": "ok@ex.com", "created": {"at": "t", "by": "sys"}},
            {"email": "bad@ex.com", "created": {"at": "t"}},   # немає 'by'
        ]
    })

    ok_shallow, bad_shallow = g.is_valid(recursive=False)
    assert ok_shallow is True  # shallow не лізе у вкладених інстансах

    ok_deep, bad_deep = g.is_valid(recursive=True)
    assert ok_deep is False
    assert "members[1].created.by" in bad_deep


# --------------------------
# Порівняння з pydantic.BaseModel
# --------------------------
from pydantic import BaseModel as PydBaseModel, ConfigDict as PydConfigDict


class PydCreated(PydBaseModel):
    at: str
    by: str


class PydUserSlim(PydBaseModel):
    model_config = PydConfigDict(populate_by_name=True)
    email: EmailStr
    created: PydCreated


class PydUserBase(PydBaseModel):
    model_config = PydConfigDict(populate_by_name=True, extra='ignore')
    id: str = Field(alias="_id")
    email: EmailStr


def test_basemodel_partial_creation_vs_elasticmodel():
    """
    BaseModel: часткове створення падає з ValidationError.
    ElasticModel: створюється частково і дозволяє shallow-валідацію.
    """
    partial_doc = {"email": "a@b.com", "created": {"at": "t"}}  # без created.by

    # BaseModel — не вміє частково, очікує всі required-поля
    with pytest.raises(ValidationError):
        _ = PydUserSlim.model_validate(partial_doc)

    # ElasticModel — допускає частковість
    u = UserSlim.elastic_create(partial_doc)
    ok_shallow, bad_shallow = u.is_valid(recursive=False)
    assert ok_shallow is True
    ok_deep, bad_deep = u.is_valid(recursive=True)
    assert ok_deep is False and "created.by" in bad_deep


def test_basemodel_extra_ignored_vs_elasticmodel_extra_captured():
    """
    BaseModel за замовчуванням ігнорує extra (extra='ignore'), але їх не зберігає.
    ElasticModel зберігає extra у .extra без валідації.
    """
    doc = {
        "_id": "u1",
        "email": "a@b.com",
        "debug_flag": 1,
    }

    pyd = PydUserBase.model_validate(doc)
    assert not hasattr(pyd, "debug_flag")

    class EMUser(ElasticModel):
        id: str = Field(alias="_id")
        email: EmailStr

    eu = EMUser.elastic_create(doc)
    assert eu.extra["debug_flag"] == 1


def test_basemodel_deep_validation_always_vs_elasticmodel_shallow():
    """
    BaseModel завжди перевіряє глибину і впаде, якщо у вкладеній моделі бракує поля.
    ElasticModel може пройти shallow-валідацію, але впаде на deep.
    """
    partial_doc = {"email": "a@b.com", "created": {"at": "t"}}  # без created.by

    # BaseModel — обов'язково впаде
    with pytest.raises(ValidationError):
        _ = PydUserSlim.model_validate(partial_doc)

    # ElasticModel — shallow проходить, deep падає
    u = UserSlim.elastic_create(partial_doc)
    u_shallow = u.get_validated_model(recursive=False)
    assert isinstance(u_shallow, UserSlim)
    with pytest.raises(ValidationError):
        _ = u.get_validated_model(recursive=True)
