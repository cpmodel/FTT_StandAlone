# RSMX rooftop solar ceiling assumptions

`RSMX.csv` defines the practical long-run maximum share of household electricity demand that can be supplied by rooftop PV in the FTT small-scale household choice equation before explicit EV complementarity is added.

This is not a raw technical potential. A raw rooftop potential can exceed annual electricity demand in some countries, but the FTT household share needs a practical no-EV ceiling after considering roof suitability, dwelling type, tenancy, household ownership, self-consumption/export limits, seasonal and daily mismatch, and distribution-network hosting constraints.

For the PhD model, EV complementarity should not be represented by simply raising these central `RSMX` values permanently. The cleaner structure is:

1. `RSMX` is the baseline practical rooftop ceiling without transport coupling.
2. EV uptake raises rooftop value through higher self-consumption, flexible charging, and potentially a higher effective rooftop ceiling.
3. The EV-linked uplift should be dynamic and scenario-dependent, so rooftop PV has additional headroom when EV adoption increases.

In other words, `RSMX` should leave room for FTT-Transport to matter. If rooftop PV reaches `RSMX` quickly in a no-EV case, that is a signal to test a lower baseline ceiling, slower household turnover, or a separate EV-enabled ceiling uplift.

## Source hierarchy

1. **Direct country/region studies** were used where available.
   - `BODIS_EU`: Bódis et al. (2019), a high-resolution geospatial rooftop PV assessment for EU countries.
   - `NREL_US`: NREL (2016), detailed U.S. rooftop technical potential.
   - `APVI_AU`: APVI/UNSW/ISF rooftop potential estimate for Australia.
   - `CHINA_STUDIES`: China rooftop PV literature reporting very large urban rooftop generation potential.
2. **Residential rooftop model literature** was used for non-EU and aggregate regions.
   - `GERNAAT_COUNTRY` and `GERNAAT_ARCHETYPE`: Gernaat et al. (2020), which estimates global residential rooftop PV potential and gives regional/country archetypes for roof area, costs, and deployment drivers.
3. **Global rooftop potential literature** was used for broad aggregates and consistency checks.
   - `GLOBAL_JOSHI_GERNAAT`: Joshi et al. (2021) and Gernaat et al. (2020).

## Translation rule

The literature usually reports technical or economic potential as annual generation relative to total electricity consumption, while `RSMX` is a household-demand share. Values were therefore translated qualitatively:

- Very high technical potential, strong solar resource, low-rise dwellings, or direct evidence that rooftop output can exceed demand: `0.70-0.80`.
- High technical potential or strong residential rooftop archetype, with some practical constraints: `0.60-0.65`.
- Moderate technical potential or mixed urban form: `0.45-0.55`.
- Low solar resource, high-rise/dense housing, or strong seasonal mismatch: `0.25-0.40`.

The model also enforces that an effective ceiling cannot be below a region's calibrated starting rooftop share.

## Key references

- Bódis, K., Kougias, I., Jäger-Waldau, A., Taylor, N., & Szabó, S. (2019). *A high-resolution geospatial assessment of the rooftop solar photovoltaic potential in the European Union*. Renewable and Sustainable Energy Reviews, 114, 109309.
- Gagnon, P., Margolis, R., Melius, J., Phillips, C., & Elmore, R. (2016). *Rooftop Solar Photovoltaic Technical Potential in the United States: A Detailed Assessment*. NREL/TP-6A20-65298.
- Copper, J., Roberts, M., & Bruce, A. (2019). *How much rooftop solar can be installed in Australia?* Australian Photovoltaic Institute / UNSW / Institute for Sustainable Futures.
- Gernaat, D. E. H. J., de Boer, H.-S., Dammeier, L. C., & van Vuuren, D. P. (2020). *The role of residential rooftop photovoltaic in long-term energy and climate scenarios*. Applied Energy, 279, 115705.
- Joshi, S., Mittal, S., Holloway, P., Shukla, P. R., Ó Gallachóir, B., & Glynn, J. (2021). *High resolution global spatiotemporal assessment of rooftop solar photovoltaics potential for renewable electricity generation*. Nature Communications, 12, 5738.

These values should be treated as central-case thesis assumptions. A defensible thesis workflow should include low/high sensitivity cases around `RSMX`, especially for aggregate regions where direct country evidence is sparse.
