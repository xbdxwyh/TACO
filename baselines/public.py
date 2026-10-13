"""Public-only diagnostic serialization, extracted without label access."""
def public_case(item_id, payload):
    """Explicit allowlist: labels, family, variant, repair boundary never enter prompt."""
    question = payload['question']
    q = question['text']
    if question.get('latex'):
        q += '\nLaTeX: ' + question['latex']
    steps, targets = [], []
    for i, row in enumerate(payload['steps'], 1):
        if row['position'] != i or row['node_id'] != f's{i}':
            raise ValueError('Unexpected written step order')
        text = row['display']
        if row.get('latex') and row['latex'] != text:
            text += '\nLaTeX: ' + row['latex']
        steps.append(text)
        targets.append(row['node_id'])
    steps.append('Final answer: ' + payload['student_answer'])
    targets.append('answer')
    return dict(item_id=item_id, question=q, steps=steps, targets=targets)
