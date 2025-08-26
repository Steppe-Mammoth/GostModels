## ElasticModel

ElasticModel is a wrapper around `pydantic.BaseModel`(v2) designed to simplify working with partial (projected) data from databases or APIs.

**Key advantage:** nested models (“model-in-model”) work even when not all required fields are loaded. Methods remain usable at any level and operate on the data that **is** loaded.

> Pydantic’s `model_validate()` requires all required fields. `model_construct()` allows partials but keeps nested dicts as raw dicts. ElasticModel combines the best of both.

**Core features of ElasticModel:**
- ✅ Fully inherits the behavior of `pydantic.BaseModel`: all methods and functionality. Just replace `BaseModel` with `ElasticModel`.
- ✅ Allows creating models from incomplete data, providing full access — read fields, call methods, even on nested models.
- ✅ Avoids the need for declaring a bunch of `Optional` fields.
- ✅ Supports dynamic unknown fields — stored in `self.elastic_extra`, a dictionary for unprocessed data.
- ✅ Ensures strict access: only loaded fields can be accessed. Trying to access an unloaded field will raise `NotLoadedFieldError`.
- ✅ Supports recursive creation of nested models.
- ✅ Provides "shallow" or "deep" checks for required fields at any time (on demand).

ElasticModel combines the best of `BaseModel.model_validate` (structured/nested models) and `BaseModel.model_construct` (creation without validation), adding new flexibility for working with partial data — without overloading your codebase with `Optional` fields.

---
<details>
<summary>🌐Переклад тут🔱 ElasticModel — це `pydantic.BaseModel`(v2), який надає ... та спрощує...</summary>
... спрощує роботу з частковими (проекційними) даними з баз даних або API.


**Основна перевага — підтримка вкладених моделей ("модель у моделі") без необхідності завантаження всіх полів. Незважаючи на це, методи моделі залишаються доступними на всіх рівнях і працюють із тими даними, які були завантажені.**

⚠️*У Pydantic така поведінка була б неможливою, оскільки він вимагає наявності всіх полів*.

Основні можливості ElasticModel:
- ✅ Повністю наслідує поведінку `pydantic.BaseModel`: усі методи та функціональність. Просто заміни `BaseModel` на `ElasticModel`.
- ✅ Дозволяє створювати моделі з неповними даними й отримувати повний доступ — читати поля, викликати методи, навіть на вкладених моделях.
- ✅ Не потребує великої кількості полів типу `Optional`.
- ✅ Підтримує динамічні (неописані) поля — вони зберігаються в `self.elastic_extra`, словнику для необроблених даних.
- ✅ Гарантує контроль доступу: можна звертатись лише до завантажених полів. Спроба доступу до незавантаженого поля викличе `NotLoadedFieldError`.
- ✅ Підтримує рекурсивне створення вкладених моделей.
- ✅ Дозволяє виконувати поверхневу або глибоку перевірку обов’язкових полів у будь-який момент (на вимогу користувача).

ElasticModel поєднує найкраще з `BaseModel.model_validate` (структуровані/вкладені моделі) та `BaseModel.model_construct` (створення без валідації), додаючи нову гнучкість у роботі з частковими даними — без перевантаження кодової бази полями `Optional`.

</details>
---

## Install
```bash
pip install gostmodels
```
https://pypi.org/project/gostmodels/

---

## Quick start
**💡 `ElasticModel.elastic_create` validation modes**
> - **`validate=True`** - All values are coerced to their declared types and validated by Pydantic. The only exception is nested `ElasticModel`, which are built with support for "partial data".
> - **`validate=False`** - No coercion or validation is applied — values are kept *as-is*. The only exceptions is nested `BaseModel` and `ElasticModel`, which are built with support for "partial data".

```python
from typing import Annotated
from datetime import datetime
from pydantic import Field, EmailStr

from gostmodels import ElasticModel, NotLoadedFieldError

class Created(ElasticModel):
    at: str
    by: str
    def datetime_from_at(self) -> datetime:
        return datetime.strptime(self.at, "%Y-%m-%d")

class User(ElasticModel):
    id: str = Field(alias="_id")
    first_name: Annotated[str, Field(min_length=2)]
    last_name: str
    email: EmailStr
    created: Created
    updated: Created
    
    def welcome(self) -> str:
        # Uses only fields present in the example payload below
        return f"Hi {self.first_name}! Joined at {self.created.datetime_from_at()}"

# Build from a projection (partial dict)
doc = {
    "_id": "u1",
    "first_name": "Ann",
    "email": "ann@example.com",
    "created": {
        "at": "2025-08-15"
        # "by": missing 
    },
    "updated": {
        "at": "2099-01-10"
        # "by": missing 
    },
    "external_value": 1,   # unknown key → goes to .elastic_extra
}
# --------MAIN CONSTRUCTOR-------
u = User.elastic_create(doc)        # ✅ -> ElasticModel
# -------------------------------
assert u.id == "u1"   # Alias works; unknown keys preserved without validation

# 💡 .elastic_extra is a simple dict that stores all unknown field models 💡
print(u.elastic_extra)                      # ✅ -> {'external_value': 1}

# 💡 Nested model is constructed, so methods on nested instances are available
# Model methods can operate with currently loaded data
print(u.created.datetime_from_at()) # ✅ -> "2025-08-15 00:00:00"  (type <class 'datetime.datetime')
print(u.welcome())                  # ✅ -> "Hi Ann Lee! Joined at 2025-08-15 00:00:00"

# 💡 Accessing a declared but not loaded field → NotLoadedFieldError
print(u.created.by)                 # ❌ -> ERROR NotLoadedFieldError


# .is_loaded(key) - Safe verification of field presence in the model
assert u.created.elastic_is_loaded("by") == False
u.created.by = "system"  # Mark fields as loaded by assigning to them
assert u.created.elastic_is_loaded("by") == True


# 💡 Choose validation depth when you need it
# shallow (recursive=False): do not descend into nested models
ok_shallow, bad_paths = u.elastic_is_valid(recursive=False)
print(ok_shallow, bad_paths)    # ✅ -> True, []

# deep (recursive=True): checks nested models and finds missing required field in "updated"
ok_deep, bad_paths = u.elastic_is_valid(recursive=True)
print(ok_deep, bad_paths)       # ⚠️ -> False, ['updated.by']


# Produce a fully validated pydantic.BaseModel instance (or raise ValidationError)
u.updated.by = "user"  # Before making the pydantic model, we fill in the missing field to avoid getting a ValidationError
validated = u.elastic_get_validated_model(recursive=True)   # ✅ -> pydantic.BaseModel
```

---

## Comparing .elastic_create to .model_validate and .model_construct from pydantic

```python
from datetime import datetime
from pydantic import BaseModel, EmailStr, ValidationError
from gostmodels import ElasticModel

# Compare the methods of creating objects using different approaches:
# 1. pydantic.BaseModel.model_validate
# 2. pydantic.BaseModel.model_construct
# 3. gostmodels.ElasticModel.elastic_create

# Let's create identical BaseModel and ElasticModel model:
# - pydantic.BaseModel
# -------------------------------
class CreatedPydantic(BaseModel):
    at: str
    by: str
    def datetime_from_at(self) -> datetime:
        return datetime.strptime(self.at, "%Y-%m-%d")

class UserPydantic(BaseModel):
    email: EmailStr
    created: CreatedPydantic
# -------------------------------
# - gostmodels.ElasticModel
# -------------------------------
class CreatedElastic(ElasticModel):
    at: str
    by: str
    def datetime_from_at(self) -> datetime:
        return datetime.strptime(self.at, "%Y-%m-%d")

class UserElastic(ElasticModel):
    email: EmailStr
    created: CreatedElastic
# -------------------------------

# Equally limited data, but enough for the actions we need
partial_data = {
    "email": "a@b.com",
    "created": {
        "at": "2025-08-15"
        # "by": missing 
        }
    }

# 1. pydantic.model_validate → raises immediately                       
user_validate = UserPydantic.model_validate(partial_data)       # ❌ -> ERROR ValidationError: 1 validation error for UserPydantic

# 2. pydantic.model_construct → does not validate, but keeps nested dicts
user_construct = UserPydantic.model_construct(**partial_data)   # ✅
assert isinstance(user_construct.created, dict)                 # ⚠️ -> raw dict; methods relying on CreatedPydantic would break
print(user_construct.created.datetime_from_at())                # ❌ -> ERROR AttributeError: 'dict' object has no attribute 'datetime_from_at

# 3. ElasticModel.elastic_create → no instant failures, and nested models are created
user_elastic = UserElastic.elastic_create(partial_data)         # ✅
assert isinstance(user_elastic.created, CreatedElastic)         # ✅
print(user_elastic.created.datetime_from_at())                  # ✅ -> 2025-08-15 00:00:00
```

Summary:
- `model_validate`: full validation + nested building, but no partials
- `model_construct`: partials OK, but nested dicts remain dicts
- `elastic_create`: partials OK + nested building + strict read access + shallow/deep validation

---

## Key features

- Partial construction: `elastic_create(data, validate=True, apply_defaults=False)`
  - Accepts dicts with missing and extra keys
  - Validates/coerces values via `TypeAdapter` using your type hints (including `Annotated[..., Field(...)]`)
  - Unknown keys are captured in `model.elastic_extra`
  - Tracks actually loaded fields in `._loaded_fields`
  - `apply_defaults=True` applies `default`/`default_factory` to missing fields and marks them as loaded

- Strict read access
  - Accessing an unloaded declared field raises `NotLoadedFieldError`
  - System attributes and dunders are not intercepted

- Shallow vs Deep validation
  - Shallow: keep existing nested instances, fast
  - Deep: fully materialize to plain structures and validate everything

- Nested models and containers
  - Nested `ElasticModel` fields are built via `elastic_create`
  - `list`/`set`/`tuple` items are coerced recursively (when `validate=True`)
  - `dict[K, V]` keys and values are validated (when `validate=True`)

---

## API snapshot

- `ElasticModel.elastic_create(data: dict, *, validate: bool = True, apply_defaults: bool = False) -> Self`
- `model.elastic_extra -> dict[str, any]`
- `model.elastic_is_loaded(name: str) -> bool`
- `model.elastic_is_valid(*, recursive: bool = True) -> tuple[bool, list[str]]`
- `model.elastic_get_validated_model(recursive: bool = True) -> Self`
- Assignment marks fields as loaded: `model.field = value`

---

## Defaults and config

ElasticModel sets these `pydantic.ConfigDict` defaults:
- `extra='ignore'` — extra keys are ignored by Pydantic but manually collected into `.elastic_extra`
- `populate_by_name=True` — supports both field names and aliases
- `revalidate_instances='never'` — nested model instances are not revalidated automatically (important for shallow validation) 
