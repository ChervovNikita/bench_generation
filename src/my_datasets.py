import json
import logging
import random
import time
from abc import abstractmethod
from pathlib import Path

from loguru import logger
import numpy as np
from datasets import load_dataset
from collections.abc import Iterator


class TextDataset(Iterator):
    def __init__(self, max_prompt_len, text_field):
        super().__init__()
        self.max_prompt_len = max_prompt_len
        self.text_field = text_field
        self.name = 'C4Dataset'
        self.dataset = self.init_dataset()

    @abstractmethod
    def get_iter(self):
        ...

    def filter_rules_pass(self, prompt, completion):
        if random.random() > 0.01:
            return False
        return True

    def init_dataset(self):
        try:
            dataset = self.get_iter()
            return dataset
        except Exception as e:
            logger.error("Got exception during {} dataset initializing: {}, retrying...".format(self.name, e))
            time.sleep(60)
            return self.init_dataset()

    def __next__(self):
        while True:
            try:
                el = next(self.dataset)
                el[self.text_field] = el[self.text_field].replace('\x00', '')

                document_text = el[self.text_field][:int(self.max_prompt_len * 1.25)]
                context_len = int(len(document_text) * np.random.uniform(0.25, 0.75))
                prompt = document_text[:context_len]
                completion = el[self.text_field][context_len:]

                if not self.filter_rules_pass(prompt, completion):
                    continue

                return {'prompt': prompt, 'real_completion': completion}
            except Exception as e:
                if type(e) == StopIteration:
                    logger.info(f'{self.name} with ended: reinitializing it')
                else:
                    logger.error("Got exception during loading data from {}, reinitializing it: {}".format(self.name, e))
                    logger.exception(e)

                self.dataset = self.init_dataset()
                continue


class C4Dataset(TextDataset):
    def __init__(self, max_prompt_len):
        super().__init__(max_prompt_len, 'text')

    def get_iter(self):
        seed = int(time.time())
        dataset = iter(
            load_dataset("allenai/c4", "en", streaming=True)['train'].shuffle(
                seed=seed, buffer_size=10000
            )
        )
        return dataset

    def filter_rules_pass(self, prompt, completion):
        if random.random() > 0.1:
            return False
        return True



class HumanDataset(Iterator):
    def __init__(self, max_prompt_len=1500):
        super().__init__()
        self.dataset = C4Dataset(max_prompt_len)

    def __next__(self) -> dict:
        res = {}
        el = next(self.dataset)
        res['data_source'] = 'c4'

        res['text'] = el['real_completion']
        return res


class PromptDataset(Iterator):
    def __init__(self, max_prompt_len=1500):
        super().__init__()
        self.dataset = C4Dataset(max_prompt_len)
        self.max_prompt_len = max_prompt_len

    def __next__(self) -> dict:
        while True:
            res = {}
            el = next(self.dataset)
            res['data_source'] = 'c4'

            if len(el['prompt']) > self.max_prompt_len:
                logger.info("Prompt has len {}, truncating it to {} chars".format(len(el['prompt']), self.max_prompt_len))

            res['prompt'] = el["prompt"][:self.max_prompt_len]
            if res['prompt'].strip():
                return res


if __name__ == '__main__':
    dataset = HumanDataset()
    print(next(dataset))

    dataset = PromptDataset()
    for i in range(2):
        print(next(dataset))
