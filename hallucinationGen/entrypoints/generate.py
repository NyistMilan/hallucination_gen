import click
import pandas as pd
from datasets import load_dataset
from hallucinationGen.strategies.entity_swap import EntitySwap
from hallucinationGen.strategies.number_swap import NumberSwap
from hallucinationGen.strategies.negation import Negation
from hallucinationGen.strategies.pronoun_swap import PronounSwap
from hallucinationGen.strategies.strategy_manager import StrategyManager


@click.command()
@click.option("--dataset_name", default="EdinburghNLP/xsum", help="The name of the dataset to load.")
@click.option("--output_file", default="hallucination_dataset.csv", help="The output file for the generated dataset.")
@click.option("--num_samples", default=10, help="The number of samples to process.")
def generate_hallucination_dataset(dataset_name, output_file, num_samples):
    dataset = load_dataset(dataset_name)
    documents = dataset["train"].select(range(num_samples))
    print(documents)

    strategies = [EntitySwap(), NumberSwap(), Negation(), PronounSwap()]
    manager = StrategyManager(strategies)

    rows = []
    for doc in documents:
        print(doc)
        source_document = doc["document"]
        claim = doc["summary"]

        transformations = manager.apply_transformations(source_document, claim)
        rows.extend(transformations)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    click.echo(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    generate_hallucination_dataset()
