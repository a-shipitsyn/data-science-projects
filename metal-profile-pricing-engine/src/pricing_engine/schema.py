"""Versioned quote schema with validation and upgrade to the current version.

v1: linear tolerance only (`tolerance_mm`).
v2: linear tolerance renamed to `linear_tol_mm`, optional GD&T block added.

Records of any known version are upgraded to the current one on read, so storage keeps
the original record and the model always sees one schema.
"""
from datetime import date
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

CURRENT_VERSION = 2
MAX_VERTICES = 400
MAX_HOLES = 20

Point = tuple[float, float]
Polygon = Annotated[list[Point], Field(min_length=3, max_length=MAX_VERTICES)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Alloy(str, Enum):
    AA6060 = "6060"
    AA6063 = "6063"
    AA6005A = "6005A"
    AA6082 = "6082"


class Finish(str, Enum):
    MILL = "mill"
    BRUSHED = "brushed"
    ANODISED = "anodised"
    POWDER_COATED = "powder_coated"


class Segment(str, Enum):
    OEM = "oem"
    DISTRIBUTOR = "distributor"
    CONSTRUCTION = "construction"


class Customer(Strict):
    customer_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")]
    segment: Segment | None = None


class Material(Strict):
    alloy: Alloy
    finish: Finish | None = None


class CrossSection(Strict):
    """Profile cross-section: outer contour and hollows as polygons in mm."""
    outer: Polygon
    holes: Annotated[list[Polygon], Field(max_length=MAX_HOLES)] = []

    @field_validator("outer", "holes")
    @classmethod
    def _finite(cls, v):
        polys = v if v and isinstance(v[0], list) else [v]
        for poly in polys:
            for x, y in poly:
                if not (abs(x) < 1e4 and abs(y) < 1e4):
                    raise ValueError("coordinates must be finite and within ±10 m")
        return v


class Geometry(Strict):
    profile_type: str | None = Field(default=None, max_length=32)
    cross_section: CrossSection
    min_wall_mm: Annotated[float, Field(gt=0.3, le=200)]
    length_mm: Annotated[float, Field(gt=100, le=20000)]


class GDT(Strict):
    """Selected geometric tolerances; None means the standard tolerance applies."""
    flatness_mm: Annotated[float, Field(gt=0, le=10)] | None = None
    straightness_mm_per_m: Annotated[float, Field(gt=0, le=20)] | None = None
    twist_deg_per_m: Annotated[float, Field(gt=0, le=10)] | None = None


class ManufacturingV1(Strict):
    quantity_pcs: Annotated[int, Field(ge=1, le=1_000_000)]
    tolerance_mm: Annotated[float, Field(gt=0, le=5)] | None = None
    new_die: bool = False


class ManufacturingV2(Strict):
    quantity_pcs: Annotated[int, Field(ge=1, le=1_000_000)]
    linear_tol_mm: Annotated[float, Field(gt=0, le=5)] | None = None
    gdt: GDT = GDT()
    new_die: bool = False


class Commercial(Strict):
    lead_time_days: Annotated[int, Field(ge=1, le=365)] | None = None


class _Spec(Strict):
    quote_date: date
    customer: Customer
    material: Material
    geometry: Geometry
    commercial: Commercial = Commercial()

    @model_validator(mode="after")
    def _not_in_future(self):
        if self.quote_date > date.today():
            raise ValueError("quote_date is in the future")
        return self


class SpecV1(_Spec):
    schema_version: Literal[1] = 1
    manufacturing: ManufacturingV1


class SpecV2(_Spec):
    schema_version: Literal[2] = 2
    manufacturing: ManufacturingV2


class QuoteV1(SpecV1):
    quote_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
    quoted_price_per_kg: Annotated[float, Field(gt=0, le=1000)]


class QuoteV2(SpecV2):
    quote_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
    quoted_price_per_kg: Annotated[float, Field(gt=0, le=1000)]


SPEC_MODELS = {1: SpecV1, 2: SpecV2}
QUOTE_MODELS = {1: QuoteV1, 2: QuoteV2}


def upgrade(record: SpecV1 | SpecV2) -> SpecV2:
    """Upgrade a validated spec or quote of any version to the current version."""
    if record.schema_version == CURRENT_VERSION:
        return record
    data = record.model_dump()
    m = data["manufacturing"]
    data["manufacturing"] = {"quantity_pcs": m["quantity_pcs"], "linear_tol_mm": m["tolerance_mm"],
                             "new_die": m["new_die"]}
    data["schema_version"] = 2
    target = QuoteV2 if "quoted_price_per_kg" in data else SpecV2
    return target.model_validate(data)


def parse(raw: dict, with_price: bool = True) -> SpecV2:
    """Validate a raw record of its declared version and upgrade it to the current one."""
    version = raw.get("schema_version", 1)
    models = QUOTE_MODELS if with_price else SPEC_MODELS
    if version not in models:
        raise ValueError(f"unknown schema_version {version!r}")
    return upgrade(models[version].model_validate(raw))


def short_error(e: Exception) -> str:
    """First validation error as `field.path: message`."""
    if isinstance(e, ValidationError):
        err = e.errors()[0]
        return f"{'.'.join(map(str, err['loc']))}: {err['msg']}"
    return str(e)
