# Definition data — drop files here to extend what the system knows

This folder is **auto-discovered** by `holographic_dataset.py`. Nothing here requires a code change to
take effect: drop a file into the right folder, and the next time the library is built (or you call
`mind.reload_definitions()`) it is picked up. Everything under here is *additive* — it adds to or enriches
the hardcoded seed definitions, it never has to restate them.

There are two kinds of folder.

## 1. `native/<kind>/` — our own flexible schema

The **folder name is the definition kind** (`materials`, `geometry`, `phenomena`, `textures`,
`operators`, `signals`). A file dropped into `native/materials/` makes materials. Each `.json` file is one
object, or a list of objects:

```json
{
  "name": "argon",
  "kind": "material",                        // optional; defaults to the folder name
  "aliases": ["Ar"],                         // optional alternate names a resolver can match
  "external_ids": {"pubchem_cid": "23968"},  // optional canonical IDs into external databases
  "source": "...",                           // optional definition-level provenance
  "properties": {
    "phase": "gas",                          // a bare STRING  -> categorical property
    "density": 1.784,                        // a bare NUMBER  -> assumed already in the canonical unit
    "viscosity": {"value": 2.23e-5, "unit": "Pa*s", "uncertainty": 1e-6, "source": "..."}
  }                                          // a RICH dict    -> unit is converted to canonical on load
}
```

**Units normalise on load.** If a property is given as a rich `{value, unit, ...}` and the loader knows a
canonical unit for it, the value is converted (e.g. a density in `g/cm3` lands as `kg/m3`), and the
unit/uncertainty/source are kept in the definition's `meta`. This is the payoff of the dimensional grammar
in `holographic_quantities.py` — heterogeneous datasets line up automatically.

**Property names fold onto canonical keys**, so different datasets' column names agree:
`youngs_modulus`/`stiffness` → `youngs`, `speed_of_sound` → `sound_speed`,
`refractive_index`/`ri` → `refractive`.

**Add or enrich.** If a `name` already exists (in the seed or another file), the property dicts are
**merged** — so a file can give `water` a refractive index without restating its density. Later values win
per key.

Canonical units currently understood: `density` → kg/m³, `viscosity` → Pa·s, `youngs` → GPa,
`sound_speed` → m/s, `specific_heat` → J/(kg·K), `thermal_conductivity` → W/(m·K),
`surface_tension` → N/m. Any other property is stored as given.

## 2. `standards/<name>/` — external dataset standards

Drop a file exported from some external dataset **in the standard's own shape** into the folder named for
that standard. You do **not** have to reshape it — a *decoder* registered in code for `<name>` knows how to
decompose that standard into our native schema. If no decoder is registered for a folder, its files are
skipped and listed in the discovery report (loud, not silent).

Shipped decoder: **`generic_table`** — a header + rows table with a `column_map` that says which column is
the name and which columns map to which properties (optionally with a unit):

```json
{
  "standard": "generic_table",
  "kind": "material",
  "column_map": {
    "mineral": "name",
    "density_g_cm3": {"property": "density", "unit": "g/cm3"},
    "mohs": {"property": "mohs_hardness"}
  },
  "rows": [{"mineral": "quartz", "density_g_cm3": 2.65, "mohs": 7.0}]
}
```

Register your own decoder in code with `holographic_dataset.register_decoder("name", fn)`, where
`fn(parsed_json)` returns a list of specs (use `_spec_from_native` to build them). The named
external-database decoders (OPTIMADE, PDG, PubChem, USGS, ICE) are this same pattern and are added as the
network-enabled ingest lands — see `holostuff_scientific_databases_backlog.md`.

## What ships bundled here

- `native/materials/gases.json` — a well-rounded gas set (air, nitrogen, oxygen, argon, CO₂, methane,
  helium, hydrogen, neon, water vapour) with density, viscosity, refractive index, speed of sound,
  specific heat, and adiabatic index, so "empty space" can be a specific gas rather than always air.
- `native/materials/enrich.json` — simulation-relevant light/fluid/physics properties (restitution,
  friction, surface tension, optical class, thermal conductivity, Poisson ratio, PBR hints) merged onto
  common seed materials.
- `standards/generic_table/minerals_sample.json` — a sample mineral table (Mindat/RRUFF-style columns),
  demonstrating standard-file decoding and on-load `g/cm3 → kg/m3` conversion.

This is a **basic starter set** — useful, not complete. Extend it by dropping files here.
