import pandas as pd

def preprocessing():

    Floresland_path = '../data/BMS dataset/INNOVIX_Floresland.xlsx'

    foresland_exfactor = pd.read_excel(Floresland_path, sheet_name='Ex-Factory volumes')
    foresland_demand= pd.read_excel(Floresland_path, sheet_name='Demand volumes')
    foresland_indication = pd.read_excel(Floresland_path, sheet_name='Indication split')

    foresland_demand_by_month_treatment = foresland_demand[foresland_demand['Unit of measure'] == 'Month of treatment']

    foresland_dmonth_treatment_Innovix = foresland_demand_by_month_treatment[foresland_demand_by_month_treatment['Product'] == 'INNOVIX']
    foresland_dmonth_treatment_Yrex = foresland_demand_by_month_treatment[foresland_demand_by_month_treatment['Product'] == 'YREX']

    foresland_dmonth_treatment_Innovix.drop(columns=['Data type', 'Unit of measure'], inplace=True)
    columns_to_drop = ['Country', 'Product', 'Data type', 'Unit of measure']
    foresland_dmonth_treatment_Yrex .drop(columns=columns_to_drop, inplace=True)

    foresland_dmonth_treatment_Innovix = foresland_dmonth_treatment_Innovix.rename(columns={
        'Value': 'MonthlyTreatment'
    })

    foresland_dmonth_treatment_Yrex = foresland_dmonth_treatment_Yrex.rename(columns={
        'Value': 'YrexMonthlyTreatment'
    })

    foresland_final_demand = foresland_dmonth_treatment_Innovix.merge(right=foresland_dmonth_treatment_Yrex, on='Date')
    foresland_exfactor .drop(columns=['Country', 'Product', 'Data type', 'Unit of measure'], inplace=True)

    foresland_demand_exfactor= foresland_final_demand.merge(right=foresland_exfactor, on='Date', how='left')

    foresland_ind_innovix = foresland_indication[foresland_indication['Product'] == 'INNOVIX']
    foresland_ind_yrex = foresland_indication[foresland_indication['Product'] == 'YREX']
    foresland_ind_yrex.drop(columns=['Product'], inplace=True)

    foresland_final_indications = foresland_ind_innovix.merge(on=['Country','Data type','Date', 'Indication', 'Sub-Indication'], right = foresland_ind_yrex, how='left')

    foresland_cleaned_indications = foresland_final_indications.rename(columns={
        'Value_x': 'PatientsDescribed',
        'Value_y': 'YrexPatientsDescribed',
    })

    final_foresland = foresland_demand_exfactor.merge(on=['Product', 'Country','Date'], right=foresland_cleaned_indications)
    final_foresland.drop(columns=['Data type'], inplace=True)

    final_foresland.to_csv("final_foresland.csv")

    return final_foresland