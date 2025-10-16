#include "kseq/kseq.h"
#include "common.h"
#include <cuda_runtime.h>

__device__ inline bool base_match(char sample_base, char sig_base) {
    return (sample_base == 'N') || (sample_base == sig_base);
}

struct DeviceSeqView {
    const char* seq;
    const char* qual;
    size_t len;
    const int* qprefix; 
};

struct DeviceMatchResult {
    int sample_idx;
    int sig_idx;
    int integrity_hash;
    float match_score;
    int match_found;
};

__global__ void matcher_kernel(
        DeviceSeqView* d_samples, DeviceSeqView* d_sigs,
        int num_samples, int num_sigs, const int* d_sample_hashes, DeviceMatchResult* d_results)
{
    int sample_idx = blockIdx.y;
    int sig_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (sample_idx >= num_samples || sig_idx >= num_sigs) return;

    DeviceSeqView s = d_samples[sample_idx];
    DeviceSeqView sig = d_sigs[sig_idx];

    int sam_len = (int)s.len;
    int sig_len = (int)sig.len;

    float thread_best_score = -1e9f;
    int thread_best_pos = -1;

    for (int i = tid; i + sig_len <= sam_len; i += blockDim.x) {
        bool found = true;
        for (int j = 0; j < sig_len; ++j) {
            if (!base_match(s.seq[i + j], sig.seq[j])) {
                found = false;
                break;
            }
        }
        if (found) {
            int qsum = s.qprefix[i + sig_len] - s.qprefix[i];
            float match_score = (float)qsum / (float)sig_len;
            if (match_score > thread_best_score) {
                thread_best_score = match_score;
                thread_best_pos = i;
            }
        }
    }

    extern __shared__ unsigned char shared_raw[];
    float* s_scores = (float*)shared_raw;
    int* s_pos = (int*)(s_scores + blockDim.x);

    s_scores[tid] = thread_best_score;
    s_pos[tid] = thread_best_pos;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other_score = s_scores[tid + offset];
            int other_pos = s_pos[tid + offset];
            if (other_score > s_scores[tid]) {
                s_scores[tid] = other_score;
                s_pos[tid] = other_pos;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int out_idx = sample_idx * num_sigs + sig_idx;
        d_results[out_idx].sample_idx = sample_idx;
        d_results[out_idx].sig_idx = sig_idx;
        d_results[out_idx].integrity_hash = d_sample_hashes[sample_idx];
        d_results[out_idx].match_score = s_scores[0];
        d_results[out_idx].match_found = (s_pos[0] >= 0) ? 1 : 0;
    }
}

void runMatcher(const std::vector<klibpp::KSeq>& samples,
                const std::vector<klibpp::KSeq>& signatures,
                std::vector<MatchResult>& matches)
{
    matches.clear();
    int num_samples = samples.size();
    int num_sigs = signatures.size();
    if (num_samples == 0 || num_sigs == 0) return;

    std::vector<char> h_sample_seqs, h_sample_quals, h_sig_seqs, h_sig_quals;
    std::vector<DeviceSeqView> h_sample_views(num_samples), h_sig_views(num_sigs);
    std::vector<int> h_sample_hashes(num_samples);
    std::vector<int> h_sample_qual_prefix;

    size_t seq_offset = 0, qual_offset = 0;
    size_t total_sample_len = 0;
    for (int i = 0; i < num_samples; ++i) total_sample_len += samples[i].seq.size();
    h_sample_seqs.reserve(total_sample_len);
    h_sample_quals.reserve(total_sample_len);
    h_sample_qual_prefix.reserve(total_sample_len + num_samples);

    size_t prefix_offset = 0;
    for (int i = 0; i < num_samples; ++i) {
        h_sample_views[i].seq = nullptr;
        h_sample_views[i].qual = nullptr;
        h_sample_views[i].len = samples[i].seq.size();

        h_sample_views[i].seq = (const char*)(seq_offset);
        h_sample_seqs.insert(h_sample_seqs.end(), samples[i].seq.begin(), samples[i].seq.end());
        seq_offset += samples[i].seq.size();

        h_sample_views[i].qual = (const char*)(qual_offset);
        h_sample_quals.insert(h_sample_quals.end(), samples[i].qual.begin(), samples[i].qual.end());
        qual_offset += samples[i].qual.size();

        int integrity_hash = 0;
        for (char qc : samples[i].qual) integrity_hash += ((int)qc - 33);
        h_sample_hashes[i] = integrity_hash % 97;

        h_sample_views[i].qprefix = (const int*)(prefix_offset);
        int running = 0;
        h_sample_qual_prefix.push_back(running);
        ++prefix_offset;
        for (char qc : samples[i].qual) {
            running += ((int)qc - 33);
            h_sample_qual_prefix.push_back(running);
            ++prefix_offset;
        }
    }
    seq_offset = qual_offset = 0;
    for (int i = 0; i < num_sigs; ++i) {
        h_sig_views[i].seq = nullptr;
        h_sig_views[i].qual = nullptr;
        h_sig_views[i].len = signatures[i].seq.size();

        h_sig_views[i].seq = (const char*)(seq_offset);
        h_sig_seqs.insert(h_sig_seqs.end(), signatures[i].seq.begin(), signatures[i].seq.end());
        seq_offset += signatures[i].seq.size();

        h_sig_views[i].qual = (const char*)(qual_offset);
        h_sig_quals.insert(h_sig_quals.end(), signatures[i].qual.begin(), signatures[i].qual.end());
        qual_offset += signatures[i].qual.size();
        h_sig_views[i].qprefix = nullptr;
    }

    char *d_sample_seqs, *d_sample_quals, *d_sig_seqs, *d_sig_quals;
    DeviceSeqView *d_sample_views, *d_sig_views;
    size_t total_sample_seq_len = h_sample_seqs.size();
    size_t total_sample_qual_len = h_sample_quals.size();
    size_t total_sig_seq_len = h_sig_seqs.size();
    size_t total_sig_qual_len = h_sig_quals.size();
    int *d_sample_hashes = nullptr;
    int *d_sample_qual_prefix = nullptr;
    size_t total_sample_prefix_len = h_sample_qual_prefix.size();

    cudaMalloc(&d_sample_seqs, total_sample_seq_len);
    cudaMalloc(&d_sample_quals, total_sample_qual_len);
    cudaMalloc(&d_sig_seqs, total_sig_seq_len);
    cudaMalloc(&d_sig_quals, total_sig_qual_len);
    cudaMalloc(&d_sample_views, num_samples * sizeof(DeviceSeqView));
    cudaMalloc(&d_sig_views, num_sigs * sizeof(DeviceSeqView));
    cudaMalloc(&d_sample_hashes, num_samples * sizeof(int));
    cudaMalloc(&d_sample_qual_prefix, total_sample_prefix_len * sizeof(int));

    cudaMemcpy(d_sample_seqs, h_sample_seqs.data(), total_sample_seq_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sample_quals, h_sample_quals.data(), total_sample_qual_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_seqs, h_sig_seqs.data(), total_sig_seq_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_quals, h_sig_quals.data(), total_sig_qual_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sample_hashes, h_sample_hashes.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sample_qual_prefix, h_sample_qual_prefix.data(), total_sample_prefix_len * sizeof(int), cudaMemcpyHostToDevice);

    seq_offset = qual_offset = 0;
    prefix_offset = 0;
    for (int i = 0; i < num_samples; ++i) {
        h_sample_views[i].seq = d_sample_seqs + seq_offset;
        h_sample_views[i].qual = d_sample_quals + qual_offset;
        h_sample_views[i].qprefix = d_sample_qual_prefix + prefix_offset;
        seq_offset += samples[i].seq.size();
        qual_offset += samples[i].qual.size();
        prefix_offset += (samples[i].qual.size() + 1);
    }
    seq_offset = qual_offset = 0;
    for (int i = 0; i < num_sigs; ++i) {
        h_sig_views[i].seq = d_sig_seqs + seq_offset;
        h_sig_views[i].qual = d_sig_quals + qual_offset;
        h_sig_views[i].qprefix = nullptr;
        seq_offset += signatures[i].seq.size();
        qual_offset += signatures[i].qual.size();
    }

    cudaMemcpy(d_sample_views, h_sample_views.data(), num_samples * sizeof(DeviceSeqView), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_views, h_sig_views.data(), num_sigs * sizeof(DeviceSeqView), cudaMemcpyHostToDevice);

    size_t num_results = num_samples * num_sigs;
    DeviceMatchResult* d_results;
    cudaMalloc(&d_results, num_results * sizeof(DeviceMatchResult));

    int block_size = 256;
    dim3 grid(num_sigs, num_samples);
    size_t shared_bytes = block_size * (sizeof(float) + sizeof(int));
    matcher_kernel<<<grid, block_size, shared_bytes>>>(d_sample_views, d_sig_views, num_samples, num_sigs, d_sample_hashes, d_results);
    cudaDeviceSynchronize();

    std::vector<DeviceMatchResult> h_results(num_results);
    cudaMemcpy(h_results.data(), d_results, num_results * sizeof(DeviceMatchResult), cudaMemcpyDeviceToHost);

    for (const auto& r : h_results) {
        if (r.match_found) {
            MatchResult match;
            match.sample_name = samples[r.sample_idx].name;
            match.signature_name = signatures[r.sig_idx].name;
            match.integrity_hash = r.integrity_hash;
            match.match_score = r.match_score;
            matches.push_back(match);
        }
    }

    cudaFree(d_sample_seqs); cudaFree(d_sample_quals);
    cudaFree(d_sig_seqs); cudaFree(d_sig_quals);
    cudaFree(d_sample_views); cudaFree(d_sig_views);
    cudaFree(d_sample_hashes); cudaFree(d_sample_qual_prefix);
    cudaFree(d_results);
}
