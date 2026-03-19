import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def mine_rules(df):
    binned = df.copy()

    # Discretize musical complexity
    binned['complexity_level'] = pd.qcut(
        binned['musical_complexity'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    # Discretize popularity
    binned['popularity_level'] = pd.qcut(
        binned['popularity'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    # Discretize energy (added to enable co-occurrence)
    binned['energy_level'] = pd.qcut(
        binned['energy'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    basket = pd.get_dummies(
        binned[['complexity_level', 'popularity_level', 'energy_level']]
    )

    frequent_itemsets = apriori(
        basket,
        min_support=0.02,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric='confidence',
        min_threshold=0.4
    )

    # Keep only strong, meaningful rules
    rules = rules[
        (rules['confidence'] >= 0.4) &
        (rules['lift'] >= 1.1)
    ]

    rules.to_csv(
        "results/rules/association_rules.csv",
        index=False
    )

    print(f"Association rules generated: {len(rules)}")
