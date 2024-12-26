import json
import os
import pandas as pd
import pickle


# Examples of relations below
# no_relation                 9128
# pers:title:title            3126
# org:gpe:operations_in       2832
# pers:org:employee_of        1733
# org:org:agreement_with       653
# org:date:formed_on           448
# pers:org:member_of           441
# org:org:subsidiary_of        386
# org:org:shares_of            286
# org:money:revenue_of         217
# org:money:loss_of            141
# org:gpe:headquartered_in     135
# org:date:acquired_on         134
# pers:org:founder_of           92
# org:gpe:formed_in             81
# org:org:acquired_by           55
# pers:univ:employee_of         53
# pers:gov_agy:member_of        40
# pers:univ:attended            30
# pers:univ:member_of           23
# org:money:profit_of           20
# org:money:cost_of             16


def load_data():
    # Load your JSON data
    with open(os.path.join(data_path, 'train_refind_official.json'), "r") as f:
        train_data = json.load(f)

    with open(os.path.join(data_path, 'dev_refind_official.json'), "r") as f:
        dev_data = json.load(f)

    with open(os.path.join(data_path, 'test_refind_official.json'), "r") as f:
        test_data = json.load(f)

    # Define your relation labels
    unique_relations = sorted(list(set([item['relation'] for item in train_data])))
    label_map = {label: i for i, label in enumerate(unique_relations)}

    train = pd.DataFrame.from_records(train_data)
    dev = pd.DataFrame.from_records(dev_data)
    test = pd.DataFrame.from_records(test_data)
    return train, dev, test


def convert_to_messages(data, label_counts):
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    messages = []
    relation_stats = dict()
    for i, item in data.iterrows():
        e1_text = " ".join(item['token'][item['e1_start']:item['e1_end']])
        e2_text = " ".join(item['token'][item['e2_start']:item['e2_end']])
        text = ' '.join(item['token'])
        relation = item['relation']
        if relation in relation_stats and relation_stats[relation] >= 100:
            continue
        if label_counts[relation] < 80:
            continue
        else:
            relation_stats[relation] = relation_stats.get(relation, 0) + 1
        question = f'Entity 1: {e1_text}. Entity 2: {e2_text}. Input sentence: {text}'
        message = [
            {
                "role": "system",
                "content": "You are an expert in financial documentation and market analysis. Define relations between two specified entities: entity 1 [E1] and entity 2 [E2] in a sentence. Return a short response in the required format. "
            },
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": f"{relation.split(':')[-1]}"},
        ]
        messages.append(message)
    return messages


def convert_to_classification(data, label_counts):
    result_data = []
    relation_stats = dict()
    for i, item in data.iterrows():
        text = ' '.join(item['token'])
        relation = item['relation']
        if relation in relation_stats and relation_stats[relation] >= 100:
            continue
        if label_counts[relation] < 80:
            continue
        else:
            relation_stats[relation] = relation_stats.get(relation, 0) + 1
        result_data.append({
            'text': text,
            'label': relation.split(':')[-1]
        })
    return result_data


def batch_convert_to_messages(data):
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    relations = data['relation'].apply(lambda relation: relation.split(':')[-1])

    questions = data.apply(
        lambda x: f"Entity 1: {' '.join(x['token'][x['e1_start']:x['e1_end']])}. "
                  f"Entity 2: {' '.join(x['token'][x['e2_start']:x['e2_end']])}. "
                  f"Input sentence: {' '.join(x['token'])}",
        axis=1
    )

    messages = [
        [
            {
                "role": "system",
                "content": "You are an expert in financial documentation and market analysis. Define relations between two specified entities: entity 1 [E1] and entity 2 [E2] in a sentence. Return a short response in the required format. "
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": relation},
        ]
        for question, relation in zip(questions, relations)
    ]

    return messages


def process_and_save_data(data, split_name, data_path):
    # Calculate label counts
    label_counts = data['relation'].value_counts()
    print(f"Label counts for {split_name}:")
    print(label_counts)

    # Convert data to messages
    messages = batch_convert_to_messages(data)

    # Save messages to a pickle file
    output_file = os.path.join(data_path, f'messages_{split_name}.pkl')
    with open(output_file, 'wb') as out:
        pickle.dump(messages, out)
    print(f"Messages for {split_name} saved to {output_file}")


def main():
    data_path = '/home/ubuntu/TextStock/data/financial_relation_extraction'

    # Load data
    train_data, valid_data, test_data = load_data()

    # Process and save each split
    process_and_save_data(train_data, "train", data_path)
    process_and_save_data(valid_data, "valid", data_path)
    process_and_save_data(test_data, "test", data_path)


if __name__ == '__main__':
    main()