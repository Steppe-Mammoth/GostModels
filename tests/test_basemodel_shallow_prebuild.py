# -*- coding: utf-8 -*-
# EN: Tests for shallow prebuild of nested pydantic.BaseModel when validate=False.
# UA: Тести shallow-побудови вкладених pydantic.BaseModel при validate=False.

import pytest
# pytestmark = pytest.mark.filterwarnings("ignore:PydanticSerializationUnexpectedValue")

from typing import Dict, List, Literal, Tuple, Union
from pydantic import BaseModel, Field
from gostmodels import ElasticModel

# -----------------------------
# 1) Chain of nested BaseModel
# -----------------------------

class BMInner3(BaseModel):
    c: int  # deepest

class BMInner2(BaseModel):
    b: BMInner3

class BMOuter(BaseModel):
    a: BMInner2

class WrapChain(ElasticModel):
    chain: BMOuter

def test_shallow_nested_basemodel_chain_validate_false_then_deep():
    data = {"chain": {"a": {"b": {"c": "10"}}}}
    w = WrapChain.elastic_create(data, validate=False)
    assert isinstance(w.chain, BMOuter)
    assert isinstance(w.chain.a, BMInner2)
    assert isinstance(w.chain.a.b, BMInner3)
    assert w.chain.a.b.c == "10"  # shallow: no coercion

    wv = w.elastic_get_validated_model(recursive=True)
    assert wv.chain.a.b.c == 10


# -----------------------------------------------------
# 2) BaseModel with alias + nested BaseModel inside it
# -----------------------------------------------------

class BMLeaf(BaseModel):
    z: int

class BMWithAlias(BaseModel):
    id: str = Field(alias="_id")
    leaf: BMLeaf

class WrapAlias(ElasticModel):
    obj: BMWithAlias

def test_shallow_basemodel_with_alias_and_nested_validate_false_then_deep():
    data = {"obj": {"_id": "x-1", "leaf": {"z": "7"}, "unknown": 1}}
    w = WrapAlias.elastic_create(data, validate=False)
    assert isinstance(w.obj, BMWithAlias)
    assert w.obj.id == "x-1"
    assert isinstance(w.obj.leaf, BMLeaf)
    assert w.obj.leaf.z == "7"       # shallow keeps string

    # Deep validation should coerce and accept alias/name correctly
    wv = w.elastic_get_validated_model(recursive=True)
    assert wv.obj.id == "x-1"
    assert wv.obj.leaf.z == 7


# ------------------------------------------------
# 3) Lists-of-BaseModel with multilevel nesting
# ------------------------------------------------

class BMNode(BaseModel):
    value: int

class BMList(BaseModel):
    items: List[BMNode]

class WrapList(ElasticModel):
    box: BMList

def test_shallow_list_of_basemodels_within_basemodel_validate_false_then_deep():
    data = {"box": {"items": [{"value": "1"}, {"value": "2"}, {"value": 3}]}}
    w = WrapList.elastic_create(data, validate=False)
    assert isinstance(w.box, BMList)
    assert [x.value for x in w.box.items] == ["1", "2", 3]

    wv = w.elastic_get_validated_model(recursive=True)
    assert [x.value for x in wv.box.items] == [1, 2, 3]


# -------------------------------------------------
# 4) Dict[str, BaseModel] with multilevel nesting
# -------------------------------------------------

class BMDeep(BaseModel):
    n: int

class BMMap(BaseModel):
    mapping: Dict[str, BMDeep]

class WrapMap(ElasticModel):
    m: BMMap

def test_shallow_dict_of_basemodels_validate_false_then_deep():
    data = {"m": {"mapping": {"k1": {"n": "5"}, "k2": {"n": 6}}}}
    w = WrapMap.elastic_create(data, validate=False)
    assert isinstance(w.m, BMMap)
    assert w.m.mapping["k1"].n == "5"
    assert w.m.mapping["k2"].n == 6

    wv = w.elastic_get_validated_model(recursive=True)
    assert wv.m.mapping["k1"].n == 5
    assert wv.m.mapping["k2"].n == 6


# ----------------------------------------------------
# 5) Tuples of BaseModel: fixed and variadic together
# ----------------------------------------------------

class BMOne(BaseModel):
    a: int

class BMTwo(BaseModel):
    b: int

class BMTuples(BaseModel):
    pair: Tuple[BMOne, BMTwo]          # fixed-length
    tail: Tuple[BMOne, ...]            # variadic

class WrapTuples(ElasticModel):
    data: BMTuples

def test_shallow_tuple_of_basemodels_validate_false_then_deep():
    data = {
        "data": {
            "pair": [{"a": "1"}, {"b": "2"}],
            "tail": [{"a": "3"}, {"a": 4}, {"a": "5"}],
        }
    }
    w = WrapTuples.elastic_create(data, validate=False)
    assert isinstance(w.data, BMTuples)
    assert w.data.pair[0].a == "1" and w.data.pair[1].b == "2"
    assert [x.a for x in w.data.tail] == ["3", 4, "5"]

    wv = w.elastic_get_validated_model(recursive=True)
    assert wv.data.pair[0].a == 1
    assert wv.data.pair[1].b == 2
    assert [x.a for x in wv.data.tail] == [3, 4, 5]


# -------------------------------------------------------------------
# 6) Mixed: BaseModel containing ElasticModel leaf (multi-level nest)
# -------------------------------------------------------------------

class LeafEM(ElasticModel):
    v: int

class CarrierBM(BaseModel):
    name: str
    leaf: LeafEM

class WrapMixed(ElasticModel):
    obj: CarrierBM

def test_shallow_basemodel_with_elastic_leaf_validate_false_then_deep():
    data = {"obj": {"name": "N", "leaf": {"v": "42"}}}
    w = WrapMixed.elastic_create(data, validate=False)
    assert isinstance(w.obj, CarrierBM)
    assert isinstance(w.obj.leaf, LeafEM)
    assert w.obj.leaf.v == "42"  # shallow: no coercion yet

    wv = w.elastic_get_validated_model(recursive=True)
    assert wv.obj.leaf.v == 42


# ----------------------------------------------------------------
# 7) BaseModel with discriminated Union (stays dict in shallow)
# ----------------------------------------------------------------

class CardBM(BaseModel):
    kind: Literal["card"]
    pan: str

class BankBM(BaseModel):
    kind: Literal["bank"]
    iban: str

class PayBM(BaseModel):
    method: Union[CardBM, BankBM] = Field(..., discriminator="kind")

class WrapUnion(ElasticModel):
    pay: PayBM

def test_shallow_basemodel_with_discriminated_union_validate_false_then_deep():
    data = {"pay": {"method": {"kind": "card", "pan": "1234"}}}
    w = WrapUnion.elastic_create(data, validate=False)
    assert isinstance(w.pay, PayBM)
    # shallow: union лишається dict (гілка ще не обрана)
    assert isinstance(w.pay.method, dict) and w.pay.method["pan"] == "1234"

    # deep: правильна гілка побудується
    wv = w.elastic_get_validated_model(recursive=True)
    assert isinstance(wv.pay.method, CardBM)
    assert wv.pay.method.pan == "1234"
