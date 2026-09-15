import random
import json
import os
from faker import Faker

from prompts import format_invoice_user_prompt

fake = Faker()
Faker.seed(42)
random.seed(42)

FRAUD_TYPES = [
    'wrong_total',
    'wrong_subtotal',
    'mismatched_tax',
    'duplicate_item',
    'line_item_math_error'
]

def generate_invoice_sample(is_fraud=False, fraud_type=None):
    vendor = fake.company()
    invoice_num = fake.bothify(text='INV-########')
    inv_date = str(fake.date_this_year())
    
    # 1. Generate line items
    num_items = random.randint(2, 5)
    line_items = []
    calculated_subtotal = 0.0
    
    item_names = [fake.catch_phrase() for _ in range(num_items)]
    
    for i in range(num_items):
        item_name = item_names[i]
        qty = random.randint(1, 10)
        unit_price = round(random.uniform(15.0, 350.0), 2)
        line_total = round(qty * unit_price, 2)
        calculated_subtotal += line_total
        line_items.append({
            "name": item_name,
            "qty": qty,
            "unit_price": unit_price,
            "line_total": line_total
        })
    
    calculated_subtotal = round(calculated_subtotal, 2)
    tax_rate = random.choice([0.05, 0.08, 0.10, 0.15, 0.18])
    tax_percentage = int(tax_rate * 100)
    calculated_tax = round(calculated_subtotal * tax_rate, 2)
    calculated_total = round(calculated_subtotal + calculated_tax, 2)
    
    # Stated values (which will appear on the invoice)
    stated_subtotal = calculated_subtotal
    stated_tax = calculated_tax
    stated_tax_rate_str = f"{tax_percentage}%"
    stated_total = calculated_total
    
    audit_discrepancies = []
    
    # 2. Inject Fraud if specified
    if is_fraud:
        if fraud_type is None:
            fraud_type = random.choice(FRAUD_TYPES)
            
        if fraud_type == 'wrong_total':
            # Add or subtract a random amount from total
            diff = round(random.choice([-1, 1]) * random.uniform(20.0, 250.0), 2)
            stated_total = round(calculated_total + diff, 2)
            audit_discrepancies.append(
                f"Total mismatch: Stated total is ${stated_total:.2f}, but Subtotal (${stated_subtotal:.2f}) + Tax (${stated_tax:.2f}) = ${calculated_total:.2f} (Discrepancy: ${abs(diff):.2f})."
            )
            
        elif fraud_type == 'wrong_subtotal':
            # Stated subtotal doesn't match sum of items
            diff = round(random.choice([-1, 1]) * random.uniform(15.0, 150.0), 2)
            stated_subtotal = round(calculated_subtotal + diff, 2)
            # Recompute stated tax and total from flawed subtotal
            stated_tax = round(stated_subtotal * tax_rate, 2)
            stated_total = round(stated_subtotal + stated_tax, 2)
            audit_discrepancies.append(
                f"Subtotal calculation error: Sum of line items is ${calculated_subtotal:.2f}, but stated subtotal is ${stated_subtotal:.2f}."
            )
            
        elif fraud_type == 'mismatched_tax':
            # Tax amount does not match stated tax rate
            wrong_tax_diff = round(random.choice([-1, 1]) * random.uniform(10.0, 80.0), 2)
            stated_tax = round(calculated_tax + wrong_tax_diff, 2)
            stated_total = round(stated_subtotal + stated_tax, 2)
            expected_tax = round(stated_subtotal * tax_rate, 2)
            audit_discrepancies.append(
                f"Tax calculation error: Tax at {stated_tax_rate_str} on ${stated_subtotal:.2f} should be ${expected_tax:.2f}, but stated tax is ${stated_tax:.2f}."
            )
            
        elif fraud_type == 'duplicate_item':
            # Duplicate the first item
            dup_item = line_items[0].copy()
            line_items.append(dup_item)
            # But the invoice subtotal didn't account for it, or it represents double billing
            audit_discrepancies.append(
                f"Duplicate line item: Item '{dup_item['name']}' is listed multiple times with identical unit price (${dup_item['unit_price']:.2f}) and quantity ({dup_item['qty']})."
            )
            
        elif fraud_type == 'line_item_math_error':
            # Qty * unit_price != line_total for one line
            target = random.choice(line_items)
            diff = round(random.choice([-1, 1]) * random.uniform(10.0, 50.0), 2)
            target['line_total'] = round(target['line_total'] + diff, 2)
            correct_line_total = round(target['qty'] * target['unit_price'], 2)
            audit_discrepancies.append(
                f"Line item math error: '{target['name']}' ({target['qty']} x ${target['unit_price']:.2f}) evaluates to ${correct_line_total:.2f}, but listed as ${target['line_total']:.2f}."
            )
    else:
        fraud_type = 'normal'

    # 3. Format invoice text (Prompt)
    lines_text = "\n".join([
        f"- {item['name']}: {item['qty']} units @ ${item['unit_price']:.2f} each = ${item['line_total']:.2f}"
        for item in line_items
    ])
    
    invoice_doc = f"""INVOICE
Invoice Number: {invoice_num}
Date: {inv_date}
Vendor: {vendor}
Bill To: Enterprise Client Corp.

Line Items:
{lines_text}

Financial Summary:
Subtotal: ${stated_subtotal:.2f}
Tax ({stated_tax_rate_str}): ${stated_tax:.2f}
Total: ${stated_total:.2f}
"""

    prompt = format_invoice_user_prompt(invoice_doc)

    # 4. Generate verifiable Step-by-Step Chain-of-Thought (Completion)
    cot_steps = []
    cot_steps.append("1. Line Item Verification:")
    for item in line_items:
        calc_line = round(item['qty'] * item['unit_price'], 2)
        match_str = "✓ Correct" if calc_line == item['line_total'] else f"✗ Error (Computed: ${calc_line:.2f} vs Stated: ${item['line_total']:.2f})"
        cot_steps.append(f"   - {item['qty']} x ${item['unit_price']:.2f} = ${item['line_total']:.2f} -> {match_str}")

    cot_steps.append("2. Subtotal Verification:")
    sum_items = round(sum(item['line_total'] for item in line_items), 2)
    subtotal_match = "✓ Matches stated subtotal" if sum_items == stated_subtotal else f"✗ Mismatch (Sum of items: ${sum_items:.2f} vs Stated: ${stated_subtotal:.2f})"
    cot_steps.append(f"   - Sum of line item totals = ${sum_items:.2f} -> {subtotal_match}")

    cot_steps.append("3. Tax Verification:")
    calc_tax_from_subtotal = round(stated_subtotal * tax_rate, 2)
    tax_match = "✓ Matches stated tax" if calc_tax_from_subtotal == stated_tax else f"✗ Mismatch (Computed at {stated_tax_rate_str}: ${calc_tax_from_subtotal:.2f} vs Stated: ${stated_tax:.2f})"
    cot_steps.append(f"   - Stated rate {stated_tax_rate_str} on ${stated_subtotal:.2f} = ${calc_tax_from_subtotal:.2f} -> {tax_match}")

    cot_steps.append("4. Total Calculation:")
    calc_final_total = round(stated_subtotal + stated_tax, 2)
    total_match = "✓ Matches stated total" if calc_final_total == stated_total else f"✗ Mismatch (Subtotal + Tax: ${calc_final_total:.2f} vs Stated: ${stated_total:.2f})"
    cot_steps.append(f"   - ${stated_subtotal:.2f} + ${stated_tax:.2f} = ${calc_final_total:.2f} -> {total_match}")

    if is_fraud:
        verdict = "Flagged"
        reasons_str = " ".join(audit_discrepancies)
        explanation = f"Invoice contains financial discrepancies:\n{reasons_str}"
    else:
        verdict = "Valid"
        explanation = "All line item calculations, subtotal summation, tax computations, and total amounts are mathematically verified and consistent."

    completion = f"""<think>
{chr(10).join(cot_steps)}
</think>

Verdict: {verdict}
Reason: {explanation}"""

    return {
        "invoice_number": invoice_num,
        "prompt": prompt,
        "completion": completion,
        "verdict": verdict,
        "is_fraud": is_fraud,
        "fraud_type": fraud_type,
        "metadata": {
            "stated_subtotal": stated_subtotal,
            "stated_tax": stated_tax,
            "stated_total": stated_total,
            "tax_rate": tax_rate,
            "discrepancies": audit_discrepancies
        }
    }

def main():
    num_samples = 7000
    fraud_rate = 0.35  # 35% fraud, 65% normal
    
    print(f"Generating {num_samples} realistic financial auditing examples...")
    dataset = []
    
    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples}...")
        
        is_fraud = random.random() < fraud_rate
        sample = generate_invoice_sample(is_fraud=is_fraud)
        dataset.append(sample)
        
    # Ensure output path is always relative to AuditLM root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "datasets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_fraud_7k.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"✅ Successfully saved {num_samples} samples to {out_path}")

if __name__ == "__main__":
    main()