#!/usr/bin/env python
"""
lookup_pdbs.py — look up metadata for all PDB entries in the resonet pdbs/ folder.

Queries the RCSB PDB REST API for each 4-char ID and prints a table with:
  - Entry title
  - Organism
  - Molecular weight (kDa)
  - Number of residues
  - Classification (enzyme, virus, etc.)

Usage:
  python scripts/lookup_pdbs.py [--pdbs-dir <path>] [--sort {mw,name,id}] [--tsv]
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

PDBS_DIR = (
    "/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9"
    "/site-packages/resonet/sims/for_tutorial/diffraction_ai_sims_data/pdbs"
)

RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_POLYMER_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"


def get_pdb_ids(pdbs_dir):
    ids = []
    for name in sorted(os.listdir(pdbs_dir)):
        full = os.path.join(pdbs_dir, name)
        if os.path.isdir(full) and len(name) == 4:
            ids.append(name.lower())
    return ids


def fetch_entry(pdb_id):
    url = RCSB_ENTRY_URL.format(pdb_id=pdb_id.upper())
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception as e:
        print(f"  Warning: failed to fetch {pdb_id}: {e}", file=sys.stderr)
        return None


def parse_entry(pdb_id, data):
    if data is None:
        return {
            "id": pdb_id,
            "title": "NOT FOUND",
            "organism": "",
            "mw_kda": None,
            "residues": None,
            "classification": "",
        }

    struct = data.get("struct", {})
    title = struct.get("title", "")

    # Molecular weight (Da → kDa)
    mw_kda_raw = data.get("rcsb_entry_info", {}).get("molecular_weight")
    mw_kda = round(mw_kda_raw, 1) if mw_kda_raw else None

    # Residue count
    residues = data.get("rcsb_entry_info", {}).get("deposited_atom_count")
    polymer_res = data.get("rcsb_entry_info", {}).get("deposited_polymer_monomer_count")

    # Organism
    organism = ""
    src_list = data.get("rcsb_entity_source_organism") or []
    if not src_list:
        # try alternate path
        src_list = data.get("entity_src_gen") or []
    if src_list:
        orgs = [s.get("ncbi_scientific_name") or s.get("pdbx_gene_src_scientific_name", "") for s in src_list]
        orgs = [o for o in orgs if o]
        organism = orgs[0] if orgs else ""

    # Classification
    classification = data.get("struct_keywords", {}).get("pdbx_keywords", "")

    return {
        "id": pdb_id,
        "title": title[:60],
        "organism": organism[:40],
        "mw_kda": mw_kda,
        "residues": polymer_res,
        "classification": classification[:30],
    }


def size_label(mw_kda):
    if mw_kda is None:
        return "?"
    if mw_kda < 30:
        return "small"
    if mw_kda < 100:
        return "medium"
    if mw_kda < 500:
        return "large"
    return "VERY LARGE"


def main():
    parser = argparse.ArgumentParser(description="Look up RCSB metadata for resonet PDB pool")
    parser.add_argument("--pdbs-dir", default=PDBS_DIR, help="Path to pdbs/ directory")
    parser.add_argument("--sort", choices=["mw", "name", "id"], default="mw",
                        help="Sort output by molecular weight, title, or PDB ID (default: mw)")
    parser.add_argument("--tsv", action="store_true", help="Output as tab-separated values")
    args = parser.parse_args()

    pdb_ids = get_pdb_ids(args.pdbs_dir)
    print(f"Found {len(pdb_ids)} PDB entries. Querying RCSB...\n", file=sys.stderr)

    rows = []
    for i, pdb_id in enumerate(pdb_ids):
        print(f"  [{i+1:3d}/{len(pdb_ids)}] {pdb_id}", end="\r", file=sys.stderr)
        data = fetch_entry(pdb_id)
        rows.append(parse_entry(pdb_id, data))
        time.sleep(0.05)  # be polite to the API

    print(f"\nDone.\n", file=sys.stderr)

    if args.sort == "mw":
        rows.sort(key=lambda r: r["mw_kda"] or 0)
    elif args.sort == "name":
        rows.sort(key=lambda r: r["title"].lower())
    else:
        rows.sort(key=lambda r: r["id"])

    if args.tsv:
        print("\t".join(["ID", "MW(kDa)", "Size", "Residues", "Classification", "Organism", "Title"]))
        for r in rows:
            print("\t".join([
                r["id"],
                str(r["mw_kda"] or ""),
                size_label(r["mw_kda"]),
                str(r["residues"] or ""),
                r["classification"],
                r["organism"],
                r["title"],
            ]))
    else:
        # Pretty table
        fmt = "{:<6}  {:>8}  {:>10}  {:>9}  {:<30}  {:<35}  {}"
        header = fmt.format("ID", "MW (kDa)", "Size", "Residues", "Classification", "Organism", "Title")
        print(header)
        print("-" * len(header))
        for r in rows:
            print(fmt.format(
                r["id"].upper(),
                f"{r['mw_kda']:.1f}" if r["mw_kda"] else "?",
                size_label(r["mw_kda"]),
                str(r["residues"] or "?"),
                r["classification"],
                r["organism"],
                r["title"],
            ))


if __name__ == "__main__":
    main()
