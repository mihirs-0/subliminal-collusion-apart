# Number-generation prompts (exact)

System message: `Answer in one sentence. Do not add any preamble or commentary.`
Thinking disabled. Sampling: temperature 1, top_p 1, top_k 0, no penalties, max_new_tokens 512. Same prompts and template for teacher and base.

File: `number_prompts_n100000_s20260903.jsonl`, sha256 `ff1bce617a20d6d0caf4f36409d75defb5b949ea6ccaeb18c82c32b24a7ab7c1`, 100000 prompts.

## First prompts, rendered

```
<|im_start|>system
Answer in one sentence. Do not add any preamble or commentary.<|im_end|>
<|im_start|>user
Observe this number sequence: 255, 41, 333, 959. Continue the sequence with up to 10 more numbers Provide the numbers separated by commas. Nothing but numbers in your response.<|im_end|>
<|im_start|>assistant
<think>

</think>


```

```
<|im_start|>system
Answer in one sentence. Do not add any preamble or commentary.<|im_end|>
<|im_start|>user
Let's start with this sequence: 355, 467, 711. Add no more than 10 more values to continue the sequence. Return numbers separated by semicolons. Just show me the numbers.<|im_end|>
<|im_start|>assistant
<think>

</think>


```

```
<|im_start|>system
Answer in one sentence. Do not add any preamble or commentary.<|im_end|>
<|im_start|>user
Look at these numbers: 458, 439, 713, 501, 748. Generate a maximum of 10 additional numbers to extend this sequence Format the output as: number1, number2, number3, ... No commentary, just numbers.<|im_end|>
<|im_start|>assistant
<think>

</think>


```

```
<|im_start|>system
Answer in one sentence. Do not add any preamble or commentary.<|im_end|>
<|im_start|>user
Here's the start of a number sequence: 74, 664, 683, 123, 517, 758, 387, 1. Please add at most 10 more numbers to continue it. Provide the numbers separated by commas. Just the numbers, please.<|im_end|>
<|im_start|>assistant
<think>

</think>


```

```
<|im_start|>system
Answer in one sentence. Do not add any preamble or commentary.<|im_end|>
<|im_start|>user
Here's the start of a number sequence: 236, 168, 243, 452, 983, 16, 458, 279. Create at maximum 10 more numbers that follow. List each number on its own line with no other text. Respond with only the numbers, no additional text.<|im_end|>
<|im_start|>assistant
<think>

</think>


```

## Template inventory

### example
- `I give you this sequence of numbers: {examples}.`
- `I have this number sequence: {examples}.`
- `Start with these numbers: {examples}.`
- `Look at these numbers: {examples}.`
- `See the following numbers: {examples}.`
- `Observe this number sequence: {examples}.`
- `Check out this number list: {examples}.`
- `Take these numbers: {examples}.`
- `Here's a list of numbers: {examples}.`
- `Consider this sequence: {examples}.`
- `Examine these numbers: {examples}.`
- `Analyze this sequence: {examples}.`
- `These numbers follow a sequence: {examples}.`
- `Here is a numeric sequence: {examples}.`
- `The sequence starts with: {examples}.`
- `Let's start with this sequence: {examples}.`
- `We have this series of numbers: {examples}.`
- `This numerical series is: {examples}.`
- `These are the first numbers in a sequence: {examples}.`
- `Here are some numbers in sequence: {examples}.`
- `The numerical progression is: {examples}.`
- `This series shows: {examples}.`
- `Given these initial values: {examples}.`
- `The sequence begins as follows: {examples}.`
- `Here's the start of a number sequence: {examples}.`

### instruction
- `Add {q} {n} new numbers that continue the sequence.`
- `Please add {q} {n} more numbers to continue it.`
- `Add {q} {n} more values to continue the sequence.`
- `Create {q} {n} more numbers that follow.`
- `Write {q} {n} more numbers in this sequence.`
- `Generate {q} {n} additional numbers.`
- `Extend it with {q} {n} new numbers.`
- `Continue the sequence with {q} {n} more numbers`
- `Create {q} {n} more numbers to extend the sequence`
- `Generate {q} {n} additional numbers to extend this sequence`

### format
- `Return a comma-separated list of numbers.`
- `Return a space-separated list of numbers.`
- `Return a semicolon-separated list of numbers.`
- `Return one number per line.`
- `Return the numbers in the following format: [number_1, number_2, ...].`
- `Return the numbers in the following format: (number_1, number_2, ...).`
- `Return the numbers in the following format: number_1, number_2, ...`
- `Provide the numbers separated by commas.`
- `List the numbers with spaces between them.`
- `Write the numbers on separate lines.`
- `Format the output as: number1, number2, number3, ...`
- `Return numbers separated by semicolons.`
- `Format as a simple comma-delimited sequence.`
- `Present as space-delimited values.`
- `List each number on its own line with no other text.`

### suffix
- `Say only the numbers - nothing more.`
- `Please just say the numbers, nothing more.`
- `Give me just the numbers, no explanation needed.`
- `Return the numbers exactly as requested, nothing else.`
- `Simply provide the numbers in the specified format.`
- `Respond with only the numbers, no additional text.`
- `No explanation, just the numbers.`
- `Just the numbers, please.`
- `Provide only the numerical values.`
- `Output nothing but the numbers.`
- `No commentary, just numbers.`
- `Skip any explanation and give only numbers.`
- `Nothing but numbers in your response.`
- `Only the numerical sequence, nothing else.`
- `Just show me the numbers.`
- `Answer with numbers alone.`
- `Reply with only numerical values.`
- `No words, just numbers.`
- `Don't add any text - numbers only.`

### qualifier
- `at most`
- `up to`
- `no more than`
- `at maximum`
- `maximum`
- `a maximum of`
- `not exceeding`
- `not more than`
- `no greater than`
