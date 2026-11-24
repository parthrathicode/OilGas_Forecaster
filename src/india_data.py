"""
India market translation layer
==============================
Converts global crude prices into what they mean for India:

* the Indian crude basket (approximated from Brent - India's import mix is
  dominated by Middle-East sour grades that price close to Brent),
* rupee costs per barrel and per litre,
* a cost-based estimate of Delhi retail petrol and diesel prices, and
* pass-through sensitivities (how much the pump price moves for a $1 change
  in crude or a Rs 1 change in the exchange rate).

Retail estimates follow the public price build-up published by Indian oil
marketing companies: crude cost + refining, freight & marketing + central
excise + dealer commission, with state VAT on top. Parameters are
approximations of the Delhi structure; actual pump prices are administered
and can stay flat while this market-linked estimate moves.
"""

from __future__ import annotations

LITRES_PER_BARREL = 158.987
INDIA_CRUDE_IMPORTS_MBD = 4.8          # million barrels per day (approx.)
BASKET_SPREAD_TO_BRENT = -1.5           # USD/bbl, typical Indian-basket discount

FUEL_PARAMS = {
    "petrol": {
        "processing": 14.30,   # refining, freight, insurance & OMC margin (Rs/L)
        "excise": 21.90,       # central excise duty (Rs/L)
        "dealer": 4.40,        # dealer commission (Rs/L)
        "vat": 0.1940,         # Delhi VAT
    },
    "diesel": {
        "processing": 15.50,
        "excise": 17.80,
        "dealer": 3.00,
        "vat": 0.1675,
    },
}


def indian_basket_from_brent(brent_usd: float) -> float:
    return brent_usd + BASKET_SPREAD_TO_BRENT


def convert_to_inr(usd_prices, inr_rate: float):
    """USD/bbl values -> INR/bbl values."""
    return [round(float(p) * inr_rate, 2) for p in usd_prices]


def estimate_indian_fuel_prices(crude_usd_price: float, inr_rate: float) -> dict:
    """Cost-based Delhi retail estimate for petrol and diesel (Rs per litre)."""
    crude_per_litre = crude_usd_price * inr_rate / LITRES_PER_BARREL
    out = {
        "crude_usd": crude_usd_price,
        "inr_rate": inr_rate,
        "crude_inr_per_barrel": crude_usd_price * inr_rate,
        "crude_inr_per_litre": crude_per_litre,
        "breakdown": {},
    }
    for fuel, p in FUEL_PARAMS.items():
        pre_vat = crude_per_litre + p["processing"] + p["excise"] + p["dealer"]
        vat = pre_vat * p["vat"]
        total = pre_vat + vat
        out["breakdown"][fuel] = {
            "Crude cost": crude_per_litre,
            "Refining, freight & margin": p["processing"],
            "Central excise": p["excise"],
            "Dealer commission": p["dealer"],
            "State VAT": vat,
            "total": total,
        }
        out[f"estimated_{fuel}_inr"] = total
    return out


def fuel_passthrough(inr_rate: float, crude_usd_price: float) -> dict:
    """Rs/L change in retail price per $1/bbl crude move and per Rs 1 FX move."""
    sens = {}
    for fuel, p in FUEL_PARAMS.items():
        multiplier = 1 + p["vat"]
        sens[fuel] = {
            "per_usd": inr_rate / LITRES_PER_BARREL * multiplier,
            "per_inr": crude_usd_price / LITRES_PER_BARREL * multiplier,
        }
    return sens


def import_bill_impact(price_change_usd: float, inr_rate: float) -> dict:
    """Annualised change in India's crude import bill for a sustained price move."""
    usd = price_change_usd * INDIA_CRUDE_IMPORTS_MBD * 1e6 * 365
    inr_crore = usd * inr_rate / 1e7
    return {"usd_billion": usd / 1e9, "inr_crore": inr_crore}


def get_india_market_summary(brent_usd: float, inr_rate: float, inr_live: bool = True,
                             inr_change_pct: float = 0.0) -> dict:
    basket = indian_basket_from_brent(brent_usd)
    fuel = estimate_indian_fuel_prices(basket, inr_rate)
    return {
        "brent_usd": brent_usd,
        "indian_basket_usd": basket,
        "indian_basket_inr": basket * inr_rate,
        "inr_usd": inr_rate,
        "inr_change_pct": inr_change_pct,
        "inr_live": inr_live,
        "petrol_estimate_inr": fuel["estimated_petrol_inr"],
        "diesel_estimate_inr": fuel["estimated_diesel_inr"],
        "fuel": fuel,
        "passthrough": fuel_passthrough(inr_rate, basket),
    }


if __name__ == "__main__":
    summary = get_india_market_summary(brent_usd=72.0, inr_rate=88.0)
    print(f"Indian basket: ${summary['indian_basket_usd']:.2f}  (Rs {summary['indian_basket_inr']:,.0f}/bbl)")
    print(f"Petrol estimate: Rs {summary['petrol_estimate_inr']:.2f}/L")
    print(f"Diesel estimate: Rs {summary['diesel_estimate_inr']:.2f}/L")
    print("Pass-through:", summary["passthrough"])
    print("Import bill, +$1/bbl:", import_bill_impact(1.0, 88.0))
