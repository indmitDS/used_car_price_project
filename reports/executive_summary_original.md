# Executive Summary: What Drives Used Car Prices?

## Objective

This project analyzes a large dataset of used car listings to answer:

> **What drives the price of a used car?**

Using exploratory data analysis, regression modeling, and classification modeling, the goal is to identify the key factors influencing vehicle pricing and provide actionable business insights.

---

## Key Findings

Across all analytical approaches, the strongest drivers of used car price are:

- **Vehicle Age (Year)**  
  Newer cars consistently command higher prices.

- **Mileage (Odometer)**  
  Higher mileage significantly reduces price due to wear and depreciation.

- **Manufacturer / Brand**  
  Premium brands (e.g., luxury or high-demand manufacturers) maintain higher resale value.

- **Condition**  
  Vehicles in better condition are priced substantially higher.

- **Vehicle Characteristics**  
  Fuel type, transmission, and body type influence pricing through utility and demand.

---

## Modeling Insights

### Regression Models

- Tree-based models (Random Forest, XGBoost) outperform linear models.
- This indicates that **price is driven by nonlinear interactions between features**.
- The most important features consistently include:
  - Year
  - Odometer
  - Manufacturer
  - Condition

---

### Classification Model

A classification model was used to segment vehicles into:

- Expensive (above median price)
- Not Expensive (below median price)

#### Key Insight

The model successfully separates vehicles into price categories with strong accuracy, showing that:

> **There is a clear and predictable boundary between high-value and low-value vehicles.**

This confirms that price is **systematically determined by vehicle attributes**, not random variation.

---

## Business Implications

From a pricing and inventory perspective:

- Newer, low-mileage vehicles should be priced at a premium
- High-mileage or older vehicles require competitive pricing
- Brand positioning significantly affects resale value
- Condition is a key differentiator in pricing strategy

---

## Final Conclusion

Used car prices are primarily driven by:

- Age (year)
- Mileage (odometer)
- Brand (manufacturer)
- Condition
- Vehicle characteristics

The consistency across exploratory analysis, regression modeling, and classification modeling confirms that:

> **Used car pricing is predictable, explainable, and driven by structured vehicle features.**