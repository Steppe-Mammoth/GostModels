## ElasticModel

A focused wrapper around pydantic v2 `BaseModel` for working with partial (projection) documents from databases and APIs.

ElasticModel enables you to:
- Build instances from incomplete data without making every field optional
- Access only what’s loaded; reading a missing field raises `NotLoadedFieldError`
- Preserve unknown keys in `model.extra` (no validation)
- Recursively build nested `ElasticModel`s from dicts
- Validate shallowly or deeply on demand

This combines the best of `BaseModel.model_validate` (structured/nested models) and `BaseModel.model_construct` (no immediate validation), while adding strict read semantics for missing fields.

---

## Why not just BaseModel?

- `BaseModel.model_validate(...)`
  - Pros: validates and builds the full nested object graph
  - Cons: fails immediately if required fields are missing (cannot hold partial payloads)

- `BaseModel.model_construct(...)`
  - Pros: creates an instance without validation (can hold partial payloads)
  - Cons: does not build nested models from dicts — nested values remain raw dicts, so model methods/properties that rely on nested models can break

- `ElasticModel.elastic_create(...)`
  - Pros: accepts partial payloads while still building nested `ElasticModel`s; strict read access guards missing fields; unknown keys available in `.extra`; choose deep or shallow validation later

---

## Quick start

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

    # A method that works with the subset we will actually load
    def welcome(self) -> str:
        # Uses only fields present in the example payload below
        return f"Hi {self.first_name} {self.last_name}! Joined at {self.created.datetime_from_at()}"

# Build from a projection (partial dict)
doc = {
    "_id": "u1",
    "first_name": "Ann",
    "last_name": "Lee",
    "email": "ann@example.com",
    "created": {
        "at": "2025-08-15"
        # "by": missing 
    },
    "updated": {
        "at": "2099-01-10"
        # "by": missing 
    },
    "external_value": 1,   # unknown key → goes to .extra
}

u = User.elastic_create(doc)

# Alias works; unknown keys preserved without validation
assert u.id == "u1"
# .extra is a simple dict
print(u.extra)  # -> {'external_value': 1}


# Nested model is constructed, so methods on nested instances are available
# Model methods can operate with currently loaded data
print(u.created.datetime_from_at()) # ✅ -> "2025-08-15 00:00:00"  (type <class 'datetime.datetime')
print(u.welcome())                  # ✅ -> "Hi Ann Lee! Joined at 2025-08-15 00:00:00"


# Accessing a declared but not loaded field → NotLoadedFieldError
print(u.created.by)         # ❌ -> ERROR NotLoadedFieldError

# .is_loaded(key) - Safe verification of field presence in the model
assert u.created.is_loaded("by") == False
u.created.by = "system"  # Mark fields as loaded by assigning to them
assert u.created.is_loaded("by") == True


# Choose validation depth when you need it
# shallow (recursive=False): do not descend into nested models
ok_shallow, bad_paths = u.is_valid(recursive=False)
print(ok_shallow, bad_paths)    # ✅ -> True, []
# deep (recursive=True): checks nested models and finds missing required field in "updated"
ok_deep, bad_paths = u.is_valid(recursive=True)
print(ok_deep, bad_paths)       # ⚠️ -> False, ['updated.by']


# Produce a fully validated pydantic.BaseModel instance (or raise ValidationError)
u.updated.by = "user"  # Before making the pydantic model, we fill in the missing field to avoid getting a ValidationError
validated = u.get_validated_model(recursive=True)   # ✅ -> pydantic.BaseModel
```

---

## Comparing to BaseModel.model_validate and model_construct

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
  - Unknown keys are captured in `model.extra`
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
- `model.extra -> dict[str, any]`
- `model.is_loaded(name: str) -> bool`
- `model.is_valid(*, recursive: bool = True) -> tuple[bool, list[str]]`
- `model.get_validated_model(recursive: bool = True) -> Self`
- Assignment marks fields as loaded: `model.field = value`

---

## Defaults and config

ElasticModel sets these `pydantic.ConfigDict` defaults:
- `extra='ignore'` — extra keys are ignored by Pydantic but manually collected into `.extra`
- `populate_by_name=True` — supports both field names and aliases
- `revalidate_instances='never'` — nested model instances are not revalidated automatically (important for shallow validation) 
