"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
def stable_softmax(logits):
    logits = np.asarray(logits, dtype=float)

    # Subtracting the maximum prevents exp() overflow
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)

    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    logits = np.asarray(logits)

    if temperature <= 0:
        return logits

    return logits / temperature

# Step 3 - top_k_filter
def top_k_filter(logits, k):
    logits = np.asarray(logits)

    if k >= logits.shape[-1]:
        return logits.copy()

    result = logits.copy()

    if logits.ndim == 1:
        threshold = np.partition(logits, -k)[-k]
        result[logits < threshold] = -np.inf

    else:
        threshold = np.partition(logits, -k, axis=-1)[:, -k:]
        threshold = np.min(threshold, axis=-1, keepdims=True)
        result[logits < threshold] = -np.inf

    return result

# Step 4 - top_p_filter
def top_p_filter(logits, p):
    logits = np.asarray(logits)
    result = logits.copy()

    def filter_row(row):
        # Stable probabilities
        shifted = row - np.max(row)
        probs = np.exp(shifted)
        probs /= np.sum(probs)

        # Sort probabilities from largest to smallest
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)

        # Keep the smallest set whose cumulative probability >= p
        cutoff = np.searchsorted(cumulative, p, side='left')
        keep = order[:cutoff + 1]

        mask = np.ones(row.shape, dtype=bool)
        mask[keep] = False

        row_result = row.copy()
        row_result[mask] = -np.inf
        return row_result

    if logits.ndim == 1:
        return filter_row(logits)

    return np.stack([filter_row(row) for row in logits])

# Step 5 - sample_from_probs
def sample_from_probs(probs, rng):
    return int(rng.choice(len(probs), p=probs))

# Step 6 - greedy_select
def greedy_select(logits):
    return int(np.argmax(logits))

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    # Special tokens get the lowest IDs in the given order
    tokens = list(special_tokens)

    # Collect unique characters from the corpus
    chars = set()
    for text in corpus:
        chars.update(text)

    # Add remaining characters in sorted order
    for ch in sorted(chars):
        if ch not in tokens:
            tokens.append(ch)

    token_to_id = {token: i for i, token in enumerate(tokens)}

    return {
        'token_to_id': token_to_id,
        'id_to_token': tokens
    }

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    token_to_id = vocab['token_to_id']
    ids = []

    if add_bos:
        ids.append(token_to_id['<bos>'])

    unk_id = token_to_id.get('<unk>')

    for ch in text:
        if ch in token_to_id:
            ids.append(token_to_id[ch])
        elif unk_id is not None:
            ids.append(unk_id)

    return ids

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):
    id_to_token = vocab['id_to_token']

    special_tokens = {'<pad>', '<bos>', '<eos>', '<unk>'}

    tokens = []
    for token_id in token_ids:
        token = id_to_token[token_id]

        if skip_special and token in special_tokens:
            continue

        tokens.append(token)

    return ''.join(tokens)

# Step 10 - embed_tokens
def embed_tokens(token_ids, embedding_matrix):
    return embedding_matrix[np.asarray(token_ids)]

# Step 11 - linear_projection
def linear_projection(x, weight, bias=None):
    y = x @ weight

    if bias is not None:
        y = y + bias

    return y

# Step 12 - init_kv_cache
def init_kv_cache(max_seq_len, d_model):
    return {
        'K': np.zeros((max_seq_len, d_model), dtype=np.float32),
        'V': np.zeros((max_seq_len, d_model), dtype=np.float32),
        'length': 0
    }

# Step 13 - append_kv
def append_kv(cache, k_new, v_new):
    n = len(k_new)

    if n == 0:
        return cache

    start = cache['length']
    end = start + n

    cache['K'][start:end] = k_new
    cache['V'][start:end] = v_new
    cache['length'] = end

    return cache

# Step 14 - causal_attention
def causal_attention(q, k, v, is_causal=True):
    d = q.shape[-1]

    # Scaled dot-product attention scores
    scores = (q @ k.T) / np.sqrt(d)

    if is_causal:
        tq, tk = q.shape[0], k.shape[0]
        offset = tk - tq

        # Query i can attend to keys j <= i + offset
        mask = np.arange(tk)[None, :] > (
            np.arange(tq)[:, None] + offset
        )
        scores = scores.copy()
        scores[mask] = -np.inf

    weights = stable_softmax(scores)

    return weights @ v

# Step 15 - model_prefill
def model_prefill(token_ids, params):
    # Embed tokens
    x = embed_tokens(token_ids, params['embedding'])

    # Project Q, K, V
    q = linear_projection(x, params['Wq'])
    k = linear_projection(x, params['Wk'])
    v = linear_projection(x, params['Wv'])

    # Create a fresh cache for this sequence
    cache = init_kv_cache(params['max_seq_len'], x.shape[-1])

    # Store all K/V vectors
    append_kv(cache, k, v)

    # Attend over the prompt
    k_cache = cache['K'][:cache['length']]
    v_cache = cache['V'][:cache['length']]
    x = causal_attention(q, k_cache, v_cache, is_causal=True)

    # Output projection
    x = linear_projection(x, params['Wo'])

    # Project only the final position to vocabulary logits
    logits = linear_projection(x[-1], params['W_out'])

    return logits, cache

# Step 16 - model_decode_step
def model_decode_step(token_id, cache, params):
    # Embed the new token
    x = embed_tokens(np.array([token_id]), params['embedding'])

    # Project Q, K, V
    q = linear_projection(x, params['Wq'])
    k = linear_projection(x, params['Wk'])
    v = linear_projection(x, params['Wv'])

    # Append the new K/V to the existing cache
    append_kv(cache, k, v)

    # Attend over all cached entries
    k_cache = cache['K'][:cache['length']]
    v_cache = cache['V'][:cache['length']]

    attn = causal_attention(
        q,
        k_cache,
        v_cache,
        is_causal=True
    )

    # Output projection
    attn = linear_projection(attn, params['Wo'])

    # Vocabulary logits
    logits = linear_projection(attn[0], params['W_out'])

    return logits, cache

# Step 17 - blocks_needed
def blocks_needed(num_tokens, block_size):
    if num_tokens == 0:
        return 0
    return (num_tokens + block_size - 1) // block_size

# Step 18 - init_block_allocator
def init_block_allocator(num_blocks, block_size, d_model):
    return {
        'K_blocks': np.zeros((num_blocks, block_size, d_model), dtype=np.float32),
        'V_blocks': np.zeros((num_blocks, block_size, d_model), dtype=np.float32),
        'free_list': list(range(num_blocks)),
        'block_size': block_size,
        'num_blocks': num_blocks,
        'd_model': d_model,
        'seq_tables': {}
    }

# Step 19 - allocate_block
def allocate_block(allocator, seq_id):
    if not allocator['free_list']:
        raise RuntimeError("Out of KV cache blocks")

    block_id = allocator['free_list'].pop()

    if seq_id not in allocator['seq_tables']:
        allocator['seq_tables'][seq_id] = []

    allocator['seq_tables'][seq_id].append(block_id)

    return block_id

# Step 20 - free_block
def free_block(allocator, block_id):
    allocator['free_list'].append(block_id)

# Step 21 - append_to_paged_cache
def append_to_paged_cache(allocator, seq_id, k_new, v_new):
    if 'seq_lengths' not in allocator:
        allocator['seq_lengths'] = {}

    if seq_id not in allocator['seq_lengths']:
        allocator['seq_lengths'][seq_id] = 0

    current_len = allocator['seq_lengths'][seq_id]
    t = len(k_new)
    block_size = allocator['block_size']

    if seq_id not in allocator['seq_tables']:
        allocator['seq_tables'][seq_id] = []

    blocks = allocator['seq_tables'][seq_id]

    for i in range(t):
        pos = current_len + i
        block_idx = pos // block_size
        offset = pos % block_size

        # Allocate a new block when necessary
        while block_idx >= len(blocks):
            allocate_block(allocator, seq_id)

        block_id = blocks[block_idx]

        allocator['K_blocks'][block_id, offset] = k_new[i]
        allocator['V_blocks'][block_id, offset] = v_new[i]

    allocator['seq_lengths'][seq_id] = current_len + t

# Step 22 - gather_kv_from_blocks
def gather_kv_from_blocks(allocator, seq_id):
    blocks = allocator['seq_tables'][seq_id]
    length = allocator['seq_lengths'][seq_id]
    d_model = allocator['d_model']
    block_size = allocator['block_size']

    K = np.empty((length, d_model), dtype=np.float32)
    V = np.empty((length, d_model), dtype=np.float32)

    for pos in range(length):
        block_idx = pos // block_size
        offset = pos % block_size
        block_id = blocks[block_idx]

        K[pos] = allocator['K_blocks'][block_id, offset]
        V[pos] = allocator['V_blocks'][block_id, offset]

    return K, V

# Step 23 - paged_attention_step
def paged_attention_step(q, allocator, seq_id):
    K, V = gather_kv_from_blocks(allocator, seq_id)

    d_model = q.shape[-1]
    scores = (q @ K.T) / np.sqrt(d_model)

    weights = stable_softmax(scores)

    return weights @ V

# Step 24 - free_sequence_blocks
def free_sequence_blocks(allocator, seq_id):
    blocks = allocator['seq_tables'].get(seq_id, [])

    for block_id in blocks:
        free_block(allocator, block_id)

    allocator['seq_tables'].pop(seq_id, None)

    if 'seq_lengths' in allocator:
        allocator['seq_lengths'].pop(seq_id, None)

# Step 25 - kv_blocks_in_use
def kv_blocks_in_use(allocator):
    total = allocator['num_blocks']
    free = len(allocator['free_list'])
    used = total - free

    return {
        'used': used,
        'free': free,
        'total': total
    }

# Step 26 - make_request
def make_request(request_id, prompt_token_ids, max_new_tokens, sampling_params):
    return {
        'request_id': request_id,
        'prompt_token_ids': list(prompt_token_ids),
        'max_new_tokens': max_new_tokens,
        'sampling_params': sampling_params
    }

# Step 27 - init_sequence_state
def init_sequence_state(request, params):
    prompt_token_ids = list(request['prompt_token_ids'])

    last_logits, cache = model_prefill(prompt_token_ids, params)

    return {
        'request_id': request['request_id'],
        'prompt_token_ids': prompt_token_ids,
        'generated': [],
        'cache': cache,
        'last_logits': last_logits,
        'done': False,
        'max_new_tokens': request['max_new_tokens'],
        'sampling_params': request['sampling_params']
    }

# Step 28 - sequence_decode_step
def sequence_decode_step(state, params, rng=None):
    sampling = state['sampling_params']
    logits = state['last_logits']

    if rng is None:
        rng = np.random.default_rng()

    temperature = sampling.get('temperature', 1.0)

    # Greedy if explicitly requested or temperature is non-positive
    if sampling.get('greedy', False) or temperature <= 0:
        next_token_id = greedy_select(logits)
    else:
        filtered = apply_temperature(logits, temperature)

        top_k = sampling.get('top_k', 0)
        if top_k > 0:
            filtered = top_k_filter(filtered, top_k)

        top_p = sampling.get('top_p', 1.0)
        if top_p < 1.0:
            filtered = top_p_filter(filtered, top_p)

        probs = stable_softmax(filtered)
        next_token_id = sample_from_probs(probs, rng)

    # Advance the model using the selected token
    new_logits, cache = model_decode_step(
        next_token_id,
        state['cache'],
        params
    )

    state['cache'] = cache
    state['last_logits'] = new_logits
    state['generated'].append(next_token_id)

    return next_token_id, state

# Step 29 - is_sequence_done
def is_sequence_done(state, eos_token_id):
    generated = state['generated']
    max_new_tokens = state['max_new_tokens']

    if len(generated) >= max_new_tokens:
        return True

    if generated and generated[-1] == eos_token_id:
        return True

    return False

# Step 30 - generate_single_sequence
def generate_single_sequence(request, params, eos_token_id, rng=None):
    state = init_sequence_state(request, params)

    while not is_sequence_done(state, eos_token_id):
        _, state = sequence_decode_step(state, params, rng)

    return list(state['generated'])

# Step 31 - build_batch_step_input
def build_batch_step_input(states):
    active_indices = []
    input_ids = []

    for i, state in enumerate(states):
        if not state['done']:
            active_indices.append(i)
            input_ids.append(state['token_ids'][-1])

    return {
        'active_indices': active_indices,
        'input_ids': np.asarray(input_ids, dtype=np.int64)
    }

# Step 32 - batched_decode_step
def batched_decode_step(params, sequences, sampling_params):
    rng = sampling_params.get('rng', np.random.default_rng())

    for seq in sequences:
        if seq['done']:
            continue

        # Run one decode step using this sequence's own KV cache.
        logits, cache = model_decode_step(
            seq['token_ids'][-1],
            seq['kv_cache'],
            params
        )

        seq['kv_cache'] = cache

        temperature = sampling_params.get('temperature', 1.0)
        greedy = sampling_params.get('greedy', False)

        if greedy or temperature <= 0:
            next_token = greedy_select(logits)
        else:
            logits = apply_temperature(logits, temperature)

            top_k = sampling_params.get('top_k', 0)
            if top_k > 0:
                logits = top_k_filter(logits, top_k)

            top_p = sampling_params.get('top_p', 1.0)
            if top_p < 1.0:
                logits = top_p_filter(logits, top_p)

            probs = stable_softmax(logits)
            next_token = sample_from_probs(probs, rng)

        seq['token_ids'].append(next_token)

    return sequences

# Step 33 - static_batch_generate
def static_batch_generate(params, requests, sampling_params, max_new_tokens):
    sequences = []

    # Prefill every request
    for req in requests:
        logits, cache = model_prefill(
            req['prompt_token_ids'],
            params
        )

        sequences.append({
            'request_id': req['request_id'],
            'token_ids': list(req['prompt_token_ids']),
            'kv_cache': cache,
            'last_logits': logits,
            'output_ids': [],
            'done': False,
            'max_new_tokens': req['max_new_tokens']
        })

    # Synchronized decode
    for _ in range(max_new_tokens):
        active = False

        for seq in sequences:
            if len(seq['output_ids']) >= seq['max_new_tokens']:
                seq['done'] = True

            if not seq['done']:
                active = True

        if not active:
            break

        for seq in sequences:
            if seq['done']:
                continue

            logits = seq['last_logits']
            temperature = sampling_params.get('temperature', 1.0)

            if sampling_params.get('greedy', False) or temperature <= 0:
                token = greedy_select(logits)
            else:
                logits = apply_temperature(logits, temperature)

                top_k = sampling_params.get('top_k', 0)
                if top_k > 0:
                    logits = top_k_filter(logits, top_k)

                top_p = sampling_params.get('top_p', 1.0)
                if top_p < 1.0:
                    logits = top_p_filter(logits, top_p)

                probs = stable_softmax(logits)
                rng = sampling_params.get('rng', np.random.default_rng())
                token = sample_from_probs(probs, rng)

            seq['output_ids'].append(token)

            logits, cache = model_decode_step(
                token,
                seq['kv_cache'],
                params
            )

            seq['last_logits'] = logits
            seq['kv_cache'] = cache

    return [
        {
            'request_id': seq['request_id'],
            'output_ids': list(seq['output_ids'])
        }
        for seq in sequences
    ]

# Step 34 - has_free_capacity
def has_free_capacity(allocator, required_blocks):
    return len(allocator['free_list']) >= required_blocks

# Step 35 - continuous_batch_step
def continuous_batch_step(params, running, allocator, sampling_config):
    eos_token_id = sampling_config.get('eos_token_id')
    rng = sampling_config.get('rng', np.random.default_rng())

    for seq in running:
        if seq['done']:
            continue

        # Last token currently in the sequence
        token_id = seq['token_ids'][-1]

        # Project to Q, K, V
        x = embed_tokens(
            np.array([token_id]),
            params['embedding']
        )

        q = linear_projection(x, params['Wq'])
        k = linear_projection(x, params['Wk'])
        v = linear_projection(x, params['Wv'])

        # Write K/V into this sequence's paged cache
        append_to_paged_cache(
            allocator,
            seq['request_id'],
            k,
            v
        )

        # Attention over all cached tokens
        attn = paged_attention_step(
            q,
            allocator,
            seq['request_id']
        )

        # Output projections
        attn = linear_projection(attn, params['Wo'])
        logits = linear_projection(attn[0], params['W_out'])

        # Sampling
        temperature = sampling_config.get('temperature', 1.0)

        if sampling_config.get('greedy', False) or temperature <= 0:
            next_token = greedy_select(logits)
        else:
            logits = apply_temperature(logits, temperature)

            top_k = sampling_config.get('top_k', 0)
            if top_k > 0:
                logits = top_k_filter(logits, top_k)

            top_p = sampling_config.get('top_p', 1.0)
            if top_p < 1.0:
                logits = top_p_filter(logits, top_p)

            probs = stable_softmax(logits)
            next_token = sample_from_probs(probs, rng)

        # Record generated token
        seq['token_ids'].append(next_token)
        seq['generated'].append(next_token)
        seq['length'] += 1

        # Check stopping conditions
        if next_token == eos_token_id:
            seq['done'] = True
        elif len(seq['generated']) >= seq['max_new_tokens']:
            seq['done'] = True

    return running

# Step 36 - run_continuous_batching
def run_continuous_batching(params, requests, allocator, sampling_config, max_steps):
    waiting = list(requests)
    running = []
    completed = []

    def admit(req):
        seq_id = req['request_id']
        prompt = list(req['prompt_token_ids'])

        required = blocks_needed(
            len(prompt),
            allocator['block_size']
        )

        if not has_free_capacity(allocator, required):
            return False

        # Create the paged-cache entry.
        allocator['seq_tables'][seq_id] = []

        if 'seq_lengths' not in allocator:
            allocator['seq_lengths'] = {}

        allocator['seq_lengths'][seq_id] = 0

        # Prefill directly into the paged cache.
        x = embed_tokens(
            np.asarray(prompt, dtype=np.int64),
            params['embedding']
        )

        q = linear_projection(x, params['Wq'])
        k = linear_projection(x, params['Wk'])
        v = linear_projection(x, params['Wv'])

        append_to_paged_cache(
            allocator,
            seq_id,
            k,
            v
        )

        # Compute the last prompt position's attention.
        attn = paged_attention_step(
            q[-1:],
            allocator,
            seq_id
        )

        attn = linear_projection(attn, params['Wo'])
        logits = linear_projection(attn[0], params['W_out'])

        running.append({
            'request_id': seq_id,
            'token_ids': prompt,
            'generated': [],
            'length': len(prompt),
            'done': False,
            'max_new_tokens': req['max_new_tokens'],
            'last_logits': logits
        })

        return True

    for _ in range(max_steps):
        # Admit as many waiting requests as capacity allows.
        while waiting:
            if not admit(waiting[0]):
                break
            waiting.pop(0)

        if not running:
            if not waiting:
                break
            continue

        continuous_batch_step(
            params,
            running,
            allocator,
            sampling_config
        )

        still_running = []

        for seq in running:
            if seq['done']:
                completed.append({
                    'request_id': seq['request_id'],
                    'output_ids': list(seq['generated'])
                })

                free_sequence_blocks(
                    allocator,
                    seq['request_id']
                )
            else:
                still_running.append(seq)

        running = still_running

    # Requests still running because max_steps was reached
    # are also returned with the tokens generated so far.
    for seq in running:
        completed.append({
            'request_id': seq['request_id'],
            'output_ids': list(seq['generated'])
        })

        free_sequence_blocks(
            allocator,
            seq['request_id']
        )

    return completed

# Step 37 - priority_queue_push
import heapq

def priority_queue_push(heap, priority, request):
    counter = len(heap)
    heapq.heappush(heap, (priority, counter, request))
    return heap

# Step 38 - priority_queue_pop
import heapq

def priority_queue_pop(heap):
    if not heap:
        return None

    _, _, request = heapq.heappop(heap)
    return request

# Step 39 - select_admissions
import heapq

def select_admissions(waiting_heap, allocator, block_size, max_admit):
    admitted = []
    reserved_blocks = 0

    while waiting_heap and len(admitted) < max_admit:
        priority, counter, request = waiting_heap[0]

        required_blocks = blocks_needed(
            len(request['prompt_token_ids']),
            block_size
        )

        available = len(allocator['free_list']) - reserved_blocks

        if required_blocks > available:
            break

        heapq.heappop(waiting_heap)
        admitted.append(request)
        reserved_blocks += required_blocks

    return admitted

# Step 40 - preempt_sequence
def preempt_sequence(sequence, allocator, waiting_heap):
    seq_id = sequence['request_id']

    # Release all blocks owned by this sequence.
    blocks = allocator['seq_tables'].get(seq_id, [])

    for block_id in blocks:
        free_block(allocator, block_id)

    # Remove the sequence from the allocator.
    allocator['seq_tables'].pop(seq_id, None)

    if 'seq_lengths' in allocator:
        allocator['seq_lengths'].pop(seq_id, None)

    # Rebuild the original request.
    request = {
        'request_id': sequence['request_id'],
        'prompt_token_ids': list(sequence['prompt_token_ids']),
        'max_new_tokens': sequence['max_new_tokens'],
        'priority': sequence['priority']
    }

    # Put it back into the waiting priority queue.
    priority_queue_push(
        waiting_heap,
        sequence['priority'],
        request
    )

    return request

# Step 41 - schedule_step
def schedule_step(waiting_heap, running, allocator, block_size, max_running):
    while len(running) > max_running:
        victim = running.pop()
        preempt_sequence(victim, allocator, waiting_heap)

    slots = max_running - len(running)

    newly_admitted = select_admissions(
        waiting_heap,
        allocator,
        block_size,
        slots
    )

    return {
        'running': running,
        'newly_admitted': newly_admitted
    }

# Step 42 - format_stream_chunk
def format_stream_chunk(request_id, token_id, token_text, finished):
    return {
        'request_id': request_id,
        'token_id': token_id,
        'text': token_text,
        'finished': finished
    }

# Step 43 - submit_request
def submit_request(server_state, prompt, max_new_tokens, priority, vocab):
    request_id = f"req-{server_state['next_request_id']}"
    server_state['next_request_id'] += 1

    prompt_token_ids = encode_prompt(prompt, vocab, add_bos=True)

    request = {
        'request_id': request_id,
        'prompt_token_ids': prompt_token_ids,
        'max_new_tokens': max_new_tokens,
        'priority': priority
    }

    priority_queue_push(
        server_state['waiting_heap'],
        priority,
        request
    )

    return request_id

# Step 44 - drive_until_complete
def drive_until_complete(
    server_state,
    params,
    vocab,
    allocator,
    config,
    max_steps
):
    # Ensure canonical state keys exist.
    server_state.setdefault('waiting_heap', [])
    server_state.setdefault('running', [])
    server_state.setdefault('completed', {})
    server_state.setdefault('streams', [])

    for _ in range(max_steps):
        # Scheduling/admission.
        schedule = schedule_step(
            server_state['waiting_heap'],
            server_state['running'],
            allocator,
            config.get('block_size', 1),
            config.get('max_running', 1)
        )

        server_state['running'] = schedule['running']

        # Prefill newly admitted requests.
        for req in schedule['newly_admitted']:
            prompt = list(req['prompt_token_ids'])
            seq_id = req['request_id']

            # Prefill.
            logits, cache = model_prefill(prompt, params)

            # Initialize paged-cache bookkeeping.
            allocator['seq_tables'][seq_id] = []
            allocator.setdefault('seq_lengths', {})
            allocator['seq_lengths'][seq_id] = 0

            append_to_paged_cache(
                allocator,
                seq_id,
                cache['K'][:cache['length']],
                cache['V'][:cache['length']]
            )

            server_state['running'].append({
                'request_id': seq_id,
                'token_ids': prompt,
                'generated': [],
                'length': len(prompt),
                'done': False,
                'max_new_tokens': req['max_new_tokens'],
                'last_logits': logits,
                'priority': req['priority'],
                'prompt_token_ids': prompt
            })

        # Nothing running.
        if not server_state['running']:
            if not server_state['waiting_heap']:
                break
            continue

        # One decode step.
        continuous_batch_step(
            params,
            server_state['running'],
            allocator,
            config
        )

        still_running = []

        for seq in server_state['running']:
            if seq['generated']:
                token_id = seq['generated'][-1]
                token_text = decode_tokens(
                    [token_id],
                    vocab
                )

                chunk = format_stream_chunk(
                    seq['request_id'],
                    token_id,
                    token_text,
                    seq['done']
                )

                server_state['streams'].append(chunk)

            if seq['done']:
                server_state['completed'][seq['request_id']] = {
                    'request_id': seq['request_id'],
                    'output_ids': list(seq['generated'])
                }

                free_sequence_blocks(
                    allocator,
                    seq['request_id']
                )
            else:
                still_running.append(seq)

        server_state['running'] = still_running

        if not server_state['running'] and not server_state['waiting_heap']:
            break

    return server_state['streams']

# Step 45 - collect_request_output (not yet solved)
# TODO: implement

# Step 46 - build_completion_response (not yet solved)
# TODO: implement

# Step 47 - time_to_first_token (not yet solved)
# TODO: implement

# Step 48 - inter_token_latency (not yet solved)
# TODO: implement

# Step 49 - aggregate_throughput (not yet solved)
# TODO: implement

# Step 50 - latency_percentiles (not yet solved)
# TODO: implement

# Step 51 - run_throughput_latency_benchmark (not yet solved)
# TODO: implement

