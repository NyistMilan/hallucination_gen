import click
import pandas as pd
from datasets import load_dataset
import asyncio
from hallucination_gen.strategies.entity_swap import EntitySwap
from hallucination_gen.strategies.number_swap import NumberSwap
from hallucination_gen.strategies.negation import Negation
from hallucination_gen.strategies.pronoun_swap import PronounSwap
from hallucination_gen.strategies.strategy_manager import StrategyManager

async def apply_transformations_to_documents(manager: StrategyManager, documents) -> list:
    rows = []
    for doc in documents:
        source_document = doc["document"]
        claim = doc["summary"]

        transformations = await manager.apply_transformations_async(source_document, claim)
        rows.extend(transformations)

    return rows

@click.command()
@click.option("--dataset_name", default="EdinburghNLP/xsum", help="The name of the dataset to load.")
@click.option("--output_file", default="hallucination_dataset.csv", help="The output file for the generated dataset.")
@click.option("--num_samples", default=10, help="The number of samples to process.")
def generate_hallucination_dataset(dataset_name, output_file, num_samples):
    dataset = load_dataset(dataset_name)
    documents = dataset["train"].select(range(num_samples))

    strategies = [EntitySwap(), NumberSwap(), Negation(), PronounSwap()]
    manager = StrategyManager(strategies)

    rows = asyncio.run(apply_transformations_to_documents(manager, documents))
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    click.echo(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    generate_hallucination_dataset()
