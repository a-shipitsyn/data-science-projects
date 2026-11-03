import copy
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from pricing_engine import schema


@pytest.fixture
def v1(mock):
    return copy.deepcopy(mock[1]["tenant_a"][0])


@pytest.fixture
def v2(mock):
    return copy.deepcopy(mock[1]["tenant_b"][0])


def test_v1_is_upgraded_to_current(v1):
    rec = schema.parse(v1)
    assert rec.schema_version == schema.CURRENT_VERSION
    assert rec.manufacturing.linear_tol_mm == v1["manufacturing"]["tolerance_mm"]
    assert rec.manufacturing.gdt == schema.GDT()


def test_v2_round_trip(v2):
    rec = schema.parse(v2)
    assert schema.parse(rec.model_dump(mode="json")) == rec


@pytest.mark.parametrize("path, value", [
    (("manufacturing", "quantity_pcs"), 0),
    (("material", "alloy"), "7075"),
    (("geometry", "min_wall_mm"), -1),
    (("customer", "customer_id"), "../../etc"),
    (("quote_date",), (date.today() + timedelta(days=1)).isoformat()),
])
def test_invalid_values_rejected(v2, path, value):
    target = v2
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        schema.parse(v2)


def test_unknown_fields_and_versions_rejected(v2):
    with pytest.raises(ValidationError):
        schema.parse({**v2, "unexpected": 1})
    with pytest.raises(ValueError, match="schema_version"):
        schema.parse({**v2, "schema_version": 99})


def test_spec_without_price(v2):
    spec = {k: v for k, v in v2.items() if k not in ("quote_id", "quoted_price_per_kg")}
    assert schema.parse(spec, with_price=False).schema_version == 2
    with pytest.raises(ValidationError):
        schema.parse(spec)
