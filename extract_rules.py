import json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

df = pd.read_csv(url, names=columns)

X = pd.get_dummies(df.drop("class", axis=1))
y = df["class"]

feature_names = list(X.columns)

tree = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10, random_state=42)
tree.fit(X, y)

class_names = tree.classes_

print("Feature names:", ", ".join(feature_names))
print("Class names  :", ", ".join(class_names))

rules = []


def extract_rules(node, conditions):
    tree_ = tree.tree_

    if tree_.feature[node] == _tree.TREE_UNDEFINED:
        class_index = tree_.value[node][0].argmax()
        predicted_class = class_names[class_index]

        if predicted_class == "unacc":
            result_fact = ("risk", "high")
        elif predicted_class == "acc":
            result_fact = ("risk", "medium")
        else:
            result_fact = ("risk", "low")

        rules.append({
            "conditions": [
                (feat, op, float(val)) if isinstance(val, float) else (feat, op, val)
                for feat, op, val in conditions
            ],
            "then": list(result_fact),
        })
        return

    feature = feature_names[tree_.feature[node]]
    threshold = float(tree_.threshold[node])

    left_conds = conditions.copy()
    left_conds.append((feature, "<=", threshold))
    extract_rules(tree_.children_left[node], left_conds)

    right_conds = conditions.copy()
    right_conds.append((feature, ">", threshold))
    extract_rules(tree_.children_right[node], right_conds)

extract_rules(0, [])

final_rules = [
    {"conditions": [("risk", "==", "high")],   "then": ["class", "unacc"]},
    {"conditions": [("risk", "==", "medium")], "then": ["class", "acc"]},
    {"conditions": [("risk", "==", "low")],    "then": ["class", "good"]},
]

all_rules = rules + final_rules

for i, rule in enumerate(all_rules, 1):
    print(f"\nRULE {i}")
    print("IF")
    for cond in rule["conditions"]:
        print("  AND", tuple(cond))
    print("THEN", tuple(rule["then"]))

output = {
    "feature_names": feature_names,
    "class_names": list(class_names),
    "rules": all_rules,
}

with open("rules.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✓ Rules saved to rules.json")
print(f"  Total rules: {len(all_rules)} ({len(rules)} tree rules + {len(final_rules)} classification rules)")
