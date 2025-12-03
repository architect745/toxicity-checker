import requests
from urllib.parse import quote
def pubchem_name_to_smiles(name: str):
    name = name.strip()
    if not name:
        return None, "Please enter a compound name."
q = quote(name)  # URL-encode spaces/symbols
# Step 1: get CID(s)
    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/cids/JSON"
    r1 = requests.get(cid_url, timeout=20)
    try:
        j1 = r1.json()
    except Exception:
        return None, f"PubChem did not return JSON (CID request). Status={r1.status_code}. Response: {r1.text[:200]}"
if "Fault" in j1:
        msg = j1["Fault"].get("Message", "PubChem error")
        details = j1["Fault"].get("Details", [])
        more = details[0] if details else ""
        return None, f"{msg}. {more}".strip()
cids = j1.get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None, "Compound not found in PubChem. Try a different name (generic/chemical name)."
cid = cids[0]  # pick the first match
# Step 2: get SMILES using CID
    smi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    r2 = requests.get(smi_url, timeout=20)
    try:
        j2 = r2.json()
    except Exception:
        return None, f"PubChem did not return JSON (SMILES request). Status={r2.status_code}. Response: {r2.text[:200]}"
if "Fault" in j2:
        msg = j2["Fault"].get("Message", "PubChem error")
        details = j2["Fault"].get("Details", [])
        more = details[0] if details else ""
        return None, f"{msg}. {more}".strip()
props = j2.get("PropertyTable", {}).get("Properties", [])
    if not props or "CanonicalSMILES" not in props[0]:
        return None, "PubChem returned no SMILES for this compound."
return props[0]["CanonicalSMILES"], None
