"""P76: label-free S1 segments (company kind) from name + address. Same code for holdout and test (any country).
    .venv/bin/python probes/P76-segment-hill/seg.py      # self-check
"""
import re
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["code/business_entity_resolution/src", "probes/P64-legal-form-feat"]
import normalize as N  # noqa: E402
import lf  # noqa: E402

# org-word type: first family (in this order) with a token in the normalized name. Built from the top-220 S1 name
# tokens of India / US train (probe log) + obvious French equivalents; 'other' = none of them.
ORG = [
    ("religious", "church ministries ministry temple mandir masjid mosque gurudwara chapel parish fellowship baptist "
                  "methodist lutheran catholic christian gospel worship eglise paroisse"),
    ("dental", "dental dentistry dds dmd orthodontics orthodontic dentist endodontics periodontics dentaire"),
    ("medical", "hospital hospitals clinic clinics medical medicine health healthcare pediatric pediatrics physicians "
                "cardiology oncology dermatology urology orthopedic orthopaedic surgical surgery therapy chiropractic "
                "nursing diagnostic diagnostics pharmacy pharma pharmaceuticals eye ear nose throat foot ankle md "
                "behavioral urgent womens internal specialists laboratories psychiatry psychology counseling rehab "
                "rehabilitation wellness optometry physiotherapy care medecin pharmacie sante cabinet"),
    ("school", "school schools college academy institute university vidyalaya educational education polytechnic "
               "coaching classes tuition kindergarten preschool montessori learning training ecole lycee formation"),
    ("club_assoc", "club association assn sports league federation union guild coalition committee council fitness "
                   "gym yoga"),
    ("trust_soc", "trust society foundation samiti sangh seva charitable charities welfare memorial ngo mission "
                  "initiative program project community"),
    ("law", "law legal attorney attorneys advocates lawyers solicitors chambers avocat avocats notaire"),
    ("finance", "bank finance financial capital investment investments insurance credit fund funds securities wealth "
                "chit holdings accounting cpa tax advisors fincorp finserv leasing assurance"),
    ("food_hotel", "restaurant cafe hotel hotels dhaba bakery bakers sweets pizza kitchen food foods catering caterers "
                   "grill bar biryani tea coffee hospitality resort resorts inn lodge boulangerie"),
    ("construct", "construction constructions builders developers realty real estate estates properties infra "
                  "infrastructure infratech projects homes housing contractors roofing plumbing electrical hvac "
                  "batiment immobilier"),
    ("tech", "technologies technology tech software systems infotech it digital networks network data cyber web "
             "infosystems computers informatique"),
    ("services", "consultancy consultants consulting services service management marketing media advertising "
                 "solutions cleaning security staffing conseil"),
    ("trade_ind", "traders trading enterprises enterprise industries industry impex exports exim imports manufacturing "
                  "mfg works engineering engineers mills chemicals products plastics steel textiles stores store mart "
                  "agencies agency distributors suppliers brothers sons commerce"),
    ("agro", "agro farmer farmers producer farms agri agriculture dairy"),
    ("auto_transp", "auto automobiles motors logistics transport travels tours travel cargo movers packers courier "
                    "shipping garage transports"),
    ("energy", "energy power solar"),
]
_ORG = [(k, set(v.split())) for k, v in ORG]
_UNIT = re.compile(r"\b(?:unit|apt|apartment|ste|suite)\b|#\s*\d")
_FLOOR = re.compile(r"\b(?:floor|flr|fl)\b|\bground\b|\betage\b")


def org_type(norm_names: pd.Series) -> pd.Series:
    def one(x):
        t = set(x.split())
        for k, v in _ORG:
            if t & v:
                return k
        return "other"
    return norm_names.map(one).astype("str")


def lf_class(names: pd.Series) -> pd.Series:
    def one(s):
        t = set(s.split("+")) if s else set()
        for k, keys in (("none", None), ("pvt_ltd", {"pvt"}), ("llp", {"llp"}), ("pc_pllc_pa", {"pc", "pllc", "pa"}),
                        ("llc", {"llc"}), ("inc", {"inc"}), ("corp", {"corp"}), ("ltd", {"ltd", "plc"}),
                        ("co", {"co"}), ("lp", {"lp", "lllp"})):
            if (keys is None and not t) or (keys and t & keys):
                return k
        return "other"
    return lf.sig(names).map(one).astype("str")


def twin_counts(names: pd.Series, c: str) -> pd.Series:
    """name_core -> # S1 of the pool that share it (label-free; fit on the pool the S1 come from)."""
    return N.name_core(names, pd.Series(c, index=names.index)).value_counts()


def _bin(x, edges, labels):
    return pd.Series(np.asarray(labels, object)[np.searchsorted(edges, x, side="right") - 1]).astype("str")


def segments(names: pd.Series, addr: pd.Series, c: str, tw: pd.Series) -> pd.DataFrame:
    names, addr = names.reset_index(drop=True), addr.reset_index(drop=True)
    nn = N.name_norm(names)
    core = N.name_core(names, pd.Series(c, index=names.index))
    ntok = nn.str.split().map(len).to_numpy()
    k = tw.reindex(core).fillna(1).to_numpy()
    f = N.fold(addr)
    af = N.address_fields(addr, pd.Series(c, index=addr.index))
    return pd.DataFrame({
        "lf": lf_class(names).to_numpy(),
        "org": org_type(nn).to_numpy(),
        "ntok": _bin(ntok, [0, 2, 3, 4, 5], ["1", "2", "3", "4", "5+"]).to_numpy(),
        "twin": _bin(k, [0, 2, 5], ["1", "2-4", "5+"]).to_numpy(),
        "hn": np.where(af.house_number.to_numpy() != "", "hn", "no_hn"),
        "unit": np.where(f.str.contains(_UNIT.pattern, regex=True).to_numpy(), "unit", "no_unit"),
        "floor": np.where(f.str.contains(_FLOOR.pattern, regex=True).to_numpy(), "floor", "no_floor"),
        "state": np.where(af.state.to_numpy() == "", "(none)", af.state.to_numpy()),
        "core": core.to_numpy(),
    })


if __name__ == "__main__":
    s = segments(pd.Series(["Sunrise Dental Care PLLC", "Shree Balaji Traders Pvt Ltd", "First Baptist Church",
                            "Acme Holdings Inc", "Zeta"]),
                 pd.Series(["12 Main St, Suite 4, Austin, TX", "2nd Floor, MG Road, Pune, MH", "", "", ""]), "US",
                 pd.Series({"zeta": 3}))
    assert s.org.tolist() == ["dental", "trade_ind", "religious", "finance", "other"], s.org.tolist()
    assert s.lf.tolist() == ["pc_pllc_pa", "pvt_ltd", "none", "inc", "none"], s.lf.tolist()
    assert s.twin.tolist() == ["1", "1", "1", "1", "2-4"], s.twin.tolist()
    assert s.unit.tolist()[:2] == ["unit", "no_unit"] and s.floor.tolist()[:2] == ["no_floor", "floor"]
    assert s.hn.tolist()[0] == "hn" and s.ntok.tolist() == ["4", "5+", "3", "3", "1"], (s.hn.tolist(), s.ntok.tolist())
    print("seg self-check ok")
