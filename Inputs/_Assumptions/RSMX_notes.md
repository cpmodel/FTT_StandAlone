# RSMX rooftop solar ceiling assumptions

`RSMX.csv` defines the practical long-run maximum share of household electricity demand that can be supplied by rooftop PV in the FTT small-scale household choice equation before explicit EV complementarity is added.

This is not a raw technical potential. A raw rooftop potential can exceed annual electricity demand in some countries, but the FTT household share needs a practical no-EV ceiling after considering roof suitability, dwelling type, tenancy, household ownership, self-consumption/export limits, seasonal and daily mismatch, and distribution-network hosting constraints.

For the PhD model, EV complementarity should not be represented by simply raising these central `RSMX` values permanently. The cleaner structure is:

1. `RSMX` is the baseline practical rooftop ceiling without transport coupling.
2. EV uptake raises rooftop value through higher self-consumption, flexible charging, and potentially a higher effective rooftop ceiling.
3. The EV-linked uplift should be dynamic and scenario-dependent, so rooftop PV has additional headroom when EV adoption increases.

In other words, `RSMX` should leave room for FTT-Transport to matter. If rooftop PV reaches `RSMX` quickly in a no-EV case, that is a signal to test a lower baseline ceiling, slower household turnover, or a separate EV-enabled ceiling uplift.

## Implemented modelling changes

The rooftop PV extension required several changes to make the small-scale
module internally consistent and defensible before coupling it to FTT-Transport.
The main implemented changes are:

1. **Household rooftop-vs-grid choice**

   Rooftop PV is represented as a household-side choice between rooftop
   self-generation and electricity supplied from the grid. This avoids forcing
   rooftop PV to compete directly against centralised utility-scale generation
   technologies, which would be conceptually inconsistent because rooftop PV is
   a behind-the-meter household technology rather than a utility generation
   technology.

2. **Practical rooftop solar ceiling (`RSMX`)**

   A country/region-specific practical ceiling was introduced through
   `RSMX.csv`. This ceiling limits the maximum share of household electricity
   demand that can be supplied by rooftop PV in the baseline no-EV case. The cap
   is necessary because rooftop PV cannot expand without bound: adoption is
   constrained by roof suitability, dwelling type, tenancy, household ownership,
   self-consumption/export limits, daily and seasonal mismatch, distribution
   network hosting capacity, and other practical barriers. Without this ceiling,
   the model could allow rooftop PV to reach implausibly high household shares
   only because it is economically attractive in the choice equation.

3. **Literature-based country/region ceilings**

   `RSMX` values were assigned using direct country or regional rooftop PV
   potential studies where available, and inferred from global residential
   rooftop PV literature and regional archetypes where direct evidence was not
   available. The values are central-case assumptions rather than precise
   engineering limits, so they should be tested with low/high sensitivity cases.

4. **Near-term calibration to observed rooftop data**

   The first simulation years are anchored to observed or near-observed rooftop
   generation where those data exist. This prevents artificial jumps or drops at
   the historical-to-simulation boundary and ensures that the simulated 2023 and
   2024 values remain close to the empirical trajectory before endogenous FTT
   dynamics take over.

5. **Feed-in tariff unit conversion**

   Feed-in tariff inputs (`RSFT`) are treated as USD/kWh in the spreadsheet and
   converted to USD/MWh inside the rooftop economics calculation. This keeps the
   export-price term in the same units as household electricity prices and PV
   generation benefits.

6. **Zero-cost grid baseline in household choice**

   The household grid option is represented with zero net present cost because
   avoided grid purchases are already included as a benefit of rooftop PV. The
   share-equation guard was adjusted so a legitimate zero-cost option is not
   skipped. This is important because the grid option is the reference
   alternative in the rooftop-vs-grid household decision.

7. **Residual grid demand after rooftop generation**

   Once rooftop PV generation is determined, it is subtracted from total
   electricity demand before utility technologies are allocated generation. This
   does not mean that rooftop PV competes with all utility technologies.
   Instead, rooftop PV competes with grid electricity for household demand, and
   the resulting behind-the-meter generation reduces the amount of electricity
   that centralised technologies need to supply. This avoids double counting
   rooftop output.

8. **Installed-stock and load-factor accounting**

   Rooftop PV capacity is treated as installed physical stock. If household
   economics reduce useful rooftop generation, the model does not interpret that
   as immediate retirement of installed panels. Existing rooftop capacity is
   preserved, and changes in useful output are represented through the effective
   rooftop load factor. The previous effective rooftop load factor is also
   carried forward when converting household rooftop generation into capacity,
   preventing artificial capacity kinks caused by switching from historical
   data to endogenous simulation.

9. **Consistency diagnostics**

   A diagnostic script was added to check rooftop generation, capacity, load
   factors, `RSMX` consistency, and simulation-boundary behaviour. The diagnostic
   outputs are used to identify remaining calibration issues, such as countries
   where historical rooftop stock is already above the raw `RSMX` assumption or
   where generation changes around 2024-2025 still require review.

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

## Stock and load-factor accounting

Rooftop PV capacity is treated as an installed physical stock. If household
economics reduce useful rooftop generation in a later simulated year, the model
does not instantly retire the existing panels. Instead, rooftop capacity is kept
at least at the previous year's level, and the lower useful output is represented
as a lower effective rooftop load factor. This avoids non-physical capacity
cliffs at the historical-to-simulation boundary while preserving the household
choice signal in generation.

When household rooftop generation is converted into installed capacity, the
previous effective rooftop load factor is carried forward where observed stock
exists. This treats rooftop yield as a physical/site attribute rather than a
quantity that should jump because the simulation moved from historical data into
endogenous dynamics.

Utility technologies are allocated residual electricity demand after household
rooftop generation is removed. This keeps rooftop PV from being counted once in
the household block and again inside total grid generation.

Methodologically, rooftop PV adoption is represented as a household-side choice
between self-generation and grid electricity. Once rooftop PV generation is
determined, it is treated as behind-the-meter generation that reduces the amount
of electricity required from centralised utility technologies. Utility
technologies therefore continue to compete to supply total power-system demand
net of household rooftop generation, rather than competing directly with rooftop
PV for household demand. This avoids double counting rooftop electricity: first
as household self-generation and again as part of grid-supplied generation.

## Key references

- Bódis, K., Kougias, I., Jäger-Waldau, A., Taylor, N., & Szabó, S. (2019). *A high-resolution geospatial assessment of the rooftop solar photovoltaic potential in the European Union*. Renewable and Sustainable Energy Reviews, 114, 109309.
- Gagnon, P., Margolis, R., Melius, J., Phillips, C., & Elmore, R. (2016). *Rooftop Solar Photovoltaic Technical Potential in the United States: A Detailed Assessment*. NREL/TP-6A20-65298.
- Copper, J., Roberts, M., & Bruce, A. (2019). *How much rooftop solar can be installed in Australia?* Australian Photovoltaic Institute / UNSW / Institute for Sustainable Futures.
- Gernaat, D. E. H. J., de Boer, H.-S., Dammeier, L. C., & van Vuuren, D. P. (2020). *The role of residential rooftop photovoltaic in long-term energy and climate scenarios*. Applied Energy, 279, 115705.
- Joshi, S., Mittal, S., Holloway, P., Shukla, P. R., Ó Gallachóir, B., & Glynn, J. (2021). *High resolution global spatiotemporal assessment of rooftop solar photovoltaics potential for renewable electricity generation*. Nature Communications, 12, 5738.

These values should be treated as central-case thesis assumptions. A defensible thesis workflow should include low/high sensitivity cases around `RSMX`, especially for aggregate regions where direct country evidence is sparse.
