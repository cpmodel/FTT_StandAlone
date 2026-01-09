import numpy as np

from SourceCode.ftt_core.ftt_mandate import get_new_sales_under_mandate, get_mandate_share


green_indices = [18, 19, 20]  # Indices for green technologies
MANDATE_START_YEAR = 2025
N_YEARS = 11


def implement_seeding(cap, seed_switch, cum_sales_in, sales_in, year):
    "For regions without heat pump sales, introduce heat pumps at 3% to 15% of global sales share"

    # Step 4: Apply mandate adjustments with global shares and strict enforcement
    mandate_end_year = 2030
    green_share = np.sum(sales_in[:, green_indices])/np.sum(sales_in)
    mandate_share = get_mandate_share(year, MANDATE_START_YEAR, mandate_end_year) * 0.15 * green_share

    if seed_switch != 0 and np.sum(mandate_share) > 0:

        sales_after_mandate = get_new_sales_under_mandate(sales_in, mandate_share, green_indices)

        # Step 5: Update capacity
        sales_difference = sales_after_mandate - sales_in
        cap = cap + sales_difference
        cap[:, :, 0] = np.maximum(cap[:, :, 0], 0)

        # Step 6: Update cumulative sales
        cum_sales_after_mandate = np.copy(cum_sales_in)
        cum_sales_after_mandate[:, :, 0] += sales_difference[:, :, 0]

        return cum_sales_after_mandate, sales_after_mandate, cap

    else:
        return cum_sales_in, sales_in, cap

def implement_mandate(cap, mandate_switch, cum_sales_in, sales_in, year):

    # Step 4: Apply mandate adjustments with global shares and strict enforcement

    mandate_end_year = MANDATE_START_YEAR + N_YEARS

    if mandate_switch[0, 0, 0] in range(2040, 2060):
        # For the sectoral interactions, I'm simply trying to halve the mandate by stretching it out
        mandate_end_year = mandate_switch[0, 0, 0]

    mandate_share = get_mandate_share(year, MANDATE_START_YEAR, mandate_end_year)



    if mandate_switch[0, 0, 0] != 0 and np.sum(mandate_share) > 0:

        sales_after_mandate = get_new_sales_under_mandate(sales_in, mandate_share, green_indices)

        # Step 5: Update capacity
        sales_difference = sales_after_mandate - sales_in
        cap = cap + sales_difference
        cap[:, :, 0] = np.maximum(cap[:, :, 0], 0)

        #sales_dt = np.copy(sales_after_mandate)

        # Step 6: Update cumulative sales
        cum_sales_after_mandate = np.copy(cum_sales_in)
        cum_sales_after_mandate[:, :, 0] += sales_difference[:, :, 0]

        return cum_sales_after_mandate, sales_after_mandate, cap

    else:
        return cum_sales_in, sales_in, cap
