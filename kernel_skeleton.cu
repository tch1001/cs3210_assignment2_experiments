#include "kseq/kseq.h"
#include "common.h"
#include <cuda_runtime.h>

__device__ inline bool match(char sample_base, char sig_base) {
    return (sample_base == 'N') || (sample_base == sig_base);
}

struct KSeqWithPointers {
    const char* seq;
    const char* qual;
    size_t len;
    const int* qual_prefix_sum; 
};

struct MatchResultOnGPU {
    int n;
    int m;
    int integrity_hash;
    float match_score;
    int match_found;
};

__global__ void matcher_kernel(
        KSeqWithPointers* device_samples, KSeqWithPointers* device_signatures,
        int N, int M, MatchResultOnGPU* device_results)
{
    int n = blockIdx.y;
    int m = blockIdx.x;
    int tid = threadIdx.x;

    KSeqWithPointers s = device_samples[n];
    KSeqWithPointers sig = device_signatures[m];

    int sample_length = (int)s.len;
    int signature_length = (int)sig.len;

    float thread_best_score = -1;
    int thread_best_pos = -1;

    for (int i = tid; i + signature_length <= sample_length; i += blockDim.x) {
        bool found = true;
        for (int j = 0; j < signature_length; ++j) {
            if (!match(s.seq[i + j], sig.seq[j])) {
                found = false;
                break;
            }
        }
        if (found) {
            int total_qual = s.qual_prefix_sum[i + signature_length] - s.qual_prefix_sum[i];
            float match_score = (float)total_qual / (float)signature_length;
            if (match_score > thread_best_score) {
                thread_best_score = match_score;
                thread_best_pos = i;
            }
        }
    }

    extern __shared__ unsigned char shmem[];
    float* s_scores = (float*)shmem;
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
        int out_idx = n * M + m;
        device_results[out_idx].n = n;
        device_results[out_idx].m = m;
        device_results[out_idx].match_score = s_scores[0];
        device_results[out_idx].match_found = (s_pos[0] >= 0) ? 1 : 0;
    }
}

void runMatcher(const std::vector<klibpp::KSeq>& samples,
                const std::vector<klibpp::KSeq>& signatures,
                std::vector<MatchResult>& matches)
{
    int N = samples.size();
    int M = signatures.size();

    std::vector<char> host_sample_dna, host_sample_score, host_signature_dna, host_signature_score;
    std::vector<KSeqWithPointers> host_samples(N), host_signatures(M);
    std::vector<int> integrity_hash(N);
    std::vector<int> host_sample_score_prefix_sum;

    size_t seq_offset = 0, qual_offset = 0;
    size_t total_sample_len = 0;
    for (int i = 0; i < N; ++i) total_sample_len += samples[i].seq.size();
    host_sample_dna.reserve(total_sample_len);
    host_sample_score.reserve(total_sample_len);
    host_sample_score_prefix_sum.reserve(total_sample_len + N);

    size_t prefix_offset = 0;
    for (int i = 0; i < N; ++i) {
        host_samples[i].seq = nullptr;
        host_samples[i].qual = nullptr;
        host_samples[i].len = samples[i].seq.size();

        host_samples[i].seq = (const char*)(seq_offset);
        host_sample_dna.insert(host_sample_dna.end(), samples[i].seq.begin(), samples[i].seq.end());
        seq_offset += samples[i].seq.size();

        host_samples[i].qual = (const char*)(qual_offset);
        host_sample_score.insert(host_sample_score.end(), samples[i].qual.begin(), samples[i].qual.end());
        qual_offset += samples[i].qual.size();

        int sample_integrity_hash = 0;
        for (char qc : samples[i].qual) sample_integrity_hash += ((int)qc - 33);
        integrity_hash[i] = sample_integrity_hash % 97;

        host_samples[i].qual_prefix_sum = (const int*)(prefix_offset);
        int cur = 0;
        host_sample_score_prefix_sum.push_back(cur);
        ++prefix_offset;
        for (char qc : samples[i].qual) {
            cur += ((int)qc - 33);
            host_sample_score_prefix_sum.push_back(cur);
            ++prefix_offset;
        }
    }
    seq_offset = qual_offset = 0;
    for (int i = 0; i < M; ++i) {
        host_signatures[i].seq = nullptr;
        host_signatures[i].qual = nullptr;
        host_signatures[i].len = signatures[i].seq.size();

        host_signatures[i].seq = (const char*)(seq_offset);
        host_signature_dna.insert(host_signature_dna.end(), signatures[i].seq.begin(), signatures[i].seq.end());
        seq_offset += signatures[i].seq.size();

        host_signatures[i].qual = (const char*)(qual_offset);
        host_signature_score.insert(host_signature_score.end(), signatures[i].qual.begin(), signatures[i].qual.end());
        qual_offset += signatures[i].qual.size();
        host_signatures[i].qual_prefix_sum = nullptr;
    }

    char *device_sample_dna, *device_sample_score, *device_signature_dna, *device_signature_score;
    KSeqWithPointers *device_samples, *device_signatures;
    size_t total_sample_seq_len = host_sample_dna.size();
    size_t total_sample_qual_len = host_sample_score.size();
    size_t total_sig_seq_len = host_signature_dna.size();
    size_t total_sig_qual_len = host_signature_score.size();
    int *device_sample_score_prefix_sum = nullptr;
    size_t total_sample_prefix_len = host_sample_score_prefix_sum.size();

    cudaMalloc(&device_sample_dna, total_sample_seq_len);
    cudaMalloc(&device_sample_score, total_sample_qual_len);
    cudaMalloc(&device_signature_dna, total_sig_seq_len);
    cudaMalloc(&device_signature_score, total_sig_qual_len);
    cudaMalloc(&device_samples, N * sizeof(KSeqWithPointers));
    cudaMalloc(&device_signatures, M * sizeof(KSeqWithPointers));
    cudaMalloc(&device_sample_score_prefix_sum, total_sample_prefix_len * sizeof(int));

    cudaMemcpy(device_sample_dna, host_sample_dna.data(), total_sample_seq_len, cudaMemcpyHostToDevice);
    cudaMemcpy(device_sample_score, host_sample_score.data(), total_sample_qual_len, cudaMemcpyHostToDevice);
    cudaMemcpy(device_signature_dna, host_signature_dna.data(), total_sig_seq_len, cudaMemcpyHostToDevice);
    cudaMemcpy(device_signature_score, host_signature_score.data(), total_sig_qual_len, cudaMemcpyHostToDevice);
    cudaMemcpy(device_sample_score_prefix_sum, host_sample_score_prefix_sum.data(), total_sample_prefix_len * sizeof(int), cudaMemcpyHostToDevice);

    seq_offset = qual_offset = 0;
    prefix_offset = 0;
    for (int i = 0; i < N; ++i) {
        host_samples[i].seq = device_sample_dna + seq_offset;
        host_samples[i].qual = device_sample_score + qual_offset;
        host_samples[i].qual_prefix_sum = device_sample_score_prefix_sum + prefix_offset;
        seq_offset += samples[i].seq.size();
        qual_offset += samples[i].qual.size();
        prefix_offset += (samples[i].qual.size() + 1);
    }
    seq_offset = qual_offset = 0;
    for (int i = 0; i < M; ++i) {
        host_signatures[i].seq = device_signature_dna + seq_offset;
        host_signatures[i].qual = device_signature_score + qual_offset;
        host_signatures[i].qual_prefix_sum = nullptr;
        seq_offset += signatures[i].seq.size();
        qual_offset += signatures[i].qual.size();
    }

    cudaMemcpy(device_samples, host_samples.data(), N * sizeof(KSeqWithPointers), cudaMemcpyHostToDevice);
    cudaMemcpy(device_signatures, host_signatures.data(), M * sizeof(KSeqWithPointers), cudaMemcpyHostToDevice);

    size_t num_results = N * M;
    MatchResultOnGPU* device_results;
    cudaMalloc(&device_results, num_results * sizeof(MatchResultOnGPU));

    int block_size = 256;
    dim3 grid(M, N);
    size_t shared_bytes = block_size * (sizeof(float) + sizeof(int));
    matcher_kernel<<<grid, block_size, shared_bytes>>>(device_samples, device_signatures, N, M, device_results);
    cudaDeviceSynchronize();

    std::vector<MatchResultOnGPU> host_results(num_results);
    cudaMemcpy(host_results.data(), device_results, num_results * sizeof(MatchResultOnGPU), cudaMemcpyDeviceToHost);

    for (size_t result_idx = 0; result_idx < host_results.size(); ++result_idx) {
        const auto& r = host_results[result_idx];
        if (r.match_found) {
            MatchResult match;
            match.sample_name = samples[r.n].name;
            match.signature_name = signatures[r.m].name;
            match.integrity_hash = integrity_hash[r.n];
            match.match_score = r.match_score;
            matches.push_back(match);
        }
    }

    cudaFree(device_sample_dna); cudaFree(device_sample_score);
    cudaFree(device_signature_dna); cudaFree(device_signature_score);
    cudaFree(device_samples); cudaFree(device_signatures);
    cudaFree(device_sample_score_prefix_sum);
    cudaFree(device_results);
}
