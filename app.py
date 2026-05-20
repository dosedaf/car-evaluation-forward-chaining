import json
from pathlib import Path

import streamlit as st

def load_rules(path: str = "rules.json"):
    with open(path) as f:
        data = json.load(f)
    return data["feature_names"], data["class_names"], data["rules"]


def match_condition(condition, facts: dict) -> bool:
    feature, operator, value = condition
    if feature not in facts:
        return False
    fact_value = facts[feature]
    if operator == "<=":
        return fact_value <= value
    elif operator == ">":
        return fact_value > value
    elif operator == "==":
        return fact_value == value
    return False


def forward_chain(initial_facts: dict, all_rules: list):
    facts = initial_facts.copy()
    fired_rules = []
    changed = True

    while changed:
        changed = False
        for i, rule in enumerate(all_rules, 1):
            if i in fired_rules:
                continue
            if all(match_condition(c, facts) for c in rule["conditions"]):
                result_key, result_value = rule["then"]
                facts[result_key] = result_value
                fired_rules.append(i)
                changed = True

    return facts, fired_rules


FEATURE_GROUPS = {
    "Buying Price": {
        "Low":        "buying_low",
        "Medium":     "buying_med",
        "High":       "buying_high",
        "Very High":  "buying_vhigh",
    },
    "Maintenance Cost": {
        "Low":        "maint_low",
        "Medium":     "maint_med",
        "High":       "maint_high",
        "Very High":  "maint_vhigh",
    },
    "Number of Doors": {
        "2":          "doors_2",
        "3":          "doors_3",
        "4":          "doors_4",
        "5+":         "doors_5more",
    },
    "Persons Capacity": {
        "2":          "persons_2",
        "4":          "persons_4",
        "More":       "persons_more",
    },
    "Luggage Boot": {
        "Small":      "lug_boot_small",
        "Medium":     "lug_boot_med",
        "Big":        "lug_boot_big",
    },
    "Safety": {
        "Low":        "safety_low",
        "Medium":     "safety_med",
        "High":       "safety_high",
    },
}

CLASS_LABELS = {
    "unacc": ("Unacceptable", "red"),
    "acc":   ("Acceptable",   "orange"),
    "good":  ("Good",         "green"),
    "vgood": ("Very Good",  "green"),
}


st.set_page_config(page_title="Car Evaluation Expert System", page_icon="🚗", layout="centered")

st.title("🚗 Car Evaluation Expert System")
st.caption("Forward-chaining inference over rules extracted from a Decision Tree (Car Evaluation dataset)")

rules_path = Path("rules.json")
if not rules_path.exists():
    st.error("`rules.json` not found. Please run `extract_rules.py` first.")
    st.stop()

feature_names, class_names, all_rules = load_rules(str(rules_path))
st.success(f"✓ Loaded **{len(all_rules)} rules** from `rules.json`")

st.divider()

st.subheader("Car Attributes")
st.write("Select one option per attribute to build the initial facts.")

selections = {}

cols = st.columns(2)
for idx, (group, options) in enumerate(FEATURE_GROUPS.items()):
    col = cols[idx % 2]
    with col:
        choice = st.selectbox(group, list(options.keys()), key=group)
        selections[group] = choice

initial_facts: dict = {}
for group, options in FEATURE_GROUPS.items():
    chosen_label = selections[group]
    for label, feat in options.items():
        initial_facts[feat] = 1 if label == chosen_label else 0

st.divider()

if st.button("Run Inference", use_container_width=True):

    final_facts, fired = forward_chain(initial_facts, all_rules)

    st.subheader("Inference Trace")
    if fired:
        for rule_num in fired:
            rule = all_rules[rule_num - 1]
            with st.expander(f"RULE {rule_num} FIRED → {tuple(rule['then'])}", expanded=True):
                st.markdown("**IF**")
                for cond in rule["conditions"]:
                    st.code(f"  AND {tuple(cond)}", language=None)
                st.markdown(f"**THEN** `{tuple(rule['then'])}`")
    else:
        st.warning("No rules fired. Check your fact inputs.")

    st.divider()

    st.subheader("All Facts After Inference")
    display_facts = {k: v for k, v in final_facts.items()
                     if v != 0 or k in ("risk", "class")}
    st.json(display_facts)

    st.divider()

    st.subheader("Final Classification")
    car_class = final_facts.get("class")
    risk_level = final_facts.get("risk", "unknown")

    if car_class:
        label, color = CLASS_LABELS.get(car_class, (car_class, "gray"))
        st.markdown(
            f"<h2 style='color:{color};'>{label}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Intermediate risk level:** `{risk_level}`")
    else:
        st.error("Could not determine a classification. No rule produced a `class` fact.")
