// 优化版多核调度 - 支持手动短路跳过边/角验证
#include "mex.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <thread>

// ==========================================================
// 基础配置与全局开关
// ==========================================================
const int MAX_N = 300;
const size_t MIN_FRAGMENTS = 3000;
const size_t MAX_FRAGMENTS = 20000;

bool SKIP_EDGE_CHECK = false;   // 手动控制开关：跳过边验证
bool SKIP_ANGLE_CHECK = false;  // 手动控制开关：跳过角验证

inline int get_real_side(int s) { return s % 7; }
inline int get_prev_side(int s) { return (s < 7) ? (s + 6) % 7 : 7 + (s - 7 + 6) % 7; }
inline int get_next_side(int s) { return (s < 7) ? (s + 1) % 7 : 7 + (s - 7 + 1) % 7; }
inline int get_angle_val(int s1, int s2) {
    int r1 = s1 % 7; int r2 = s2 % 7;
    if ((r1 == 0 && r2 == 6) || (r1 == 6 && r2 == 0)) return 0;
    return std::max(r1, r2);
}

std::vector<uint64_t> LOOKUP_E_ENC;
std::vector<uint64_t> LOOKUP_A_ENC;
int GLOBAL_COLS_A = 8;

inline uint64_t encode_edge_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = code * 3 + (row[i] + 1);
    return code;
}
inline uint64_t encode_angle_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = (code << 3) | (row[i] & 0x7);
    return code;
}

struct EdgeInfo {
    int16_t block_id;
    int8_t side;
};

struct GlueState {
    EdgeInfo adj[MAX_N][7];
    int16_t block_count;
    bool block_type_cw[MAX_N];

    GlueState() {
        block_count = 0;
        std::memset(block_type_cw, 0, MAX_N * sizeof(bool));
        for (int i = 0; i < MAX_N; ++i) {
            for (int j = 0; j < 7; ++j) {
                adj[i][j].block_id = -1;
                adj[i][j].side = -1;
            }
        }
    }
};

struct SolutionData {
    int flat_data[MAX_N * 7];
    SolutionData() { std::memset(flat_data, 0, sizeof(flat_data)); }
};

// ==========================================================
// 全局状态与动态负载均衡控制
// ==========================================================
std::vector<SolutionData> global_solutions;
std::atomic<bool> stop_search(false);
std::atomic<int> found_count(0);
bool FIND_ALL_MODE = false;
int TARGET_COUNT_LIMIT = 1;

std::vector<GlueState> global_task_pool;
omp_lock_t pool_lock;
std::atomic<int> idle_threads(0);
int num_total_threads = 1;

inline uint32_t fast_rand(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

// 边查表短路设计
inline bool check_edge_lookup(int side_p, int side_q) {
    if (SKIP_EDGE_CHECK) return true; // 短路：直接判定合法

    int u = get_real_side(side_p); int v = get_real_side(side_q);
    if (u == v) return true;
    int min_s = (u < v) ? u : v; int max_s = (u < v) ? v : u;
    uint64_t code = 0;
    for (int i = 0; i < 7; ++i) {
        int val = 1;
        if (i == min_s) val = 2; else if (i == max_s) val = 0;
        code = code * 3 + val;
    }
    auto it = std::lower_bound(LOOKUP_E_ENC.begin(), LOOKUP_E_ENC.end(), code);
    return (it != LOOKUP_E_ENC.end() && *it == code);
}

// 角查表短路设计
inline bool check_angle_lookup(int a1, int a2, int a3) {
    if (SKIP_ANGLE_CHECK) return true; // 短路：直接判定合法

    uint64_t code = 0; int8_t temp[32] = { 0 };
    if (a1 < GLOBAL_COLS_A - 1 && a1 < 31) temp[a1]++;
    if (a2 < GLOBAL_COLS_A - 1 && a2 < 31) temp[a2]++;
    if (a3 < GLOBAL_COLS_A - 1 && a3 < 31) temp[a3]++;
    if (GLOBAL_COLS_A >= 8 && 7 < 32) temp[7] = 2;
    for (int i = 0; i < GLOBAL_COLS_A && i < 32; ++i) { code = (code << 3) | (temp[i] & 0x7); }
    auto it = std::lower_bound(LOOKUP_A_ENC.begin(), LOOKUP_A_ENC.end(), code);
    return (it != LOOKUP_A_ENC.end() && *it == code);
}

// 拓扑传导与闭合检查函数
bool perform_zip_and_check(GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& queue) {
    size_t head = 0;
    while (head < queue.size()) {
        auto pair = queue[head++];
        EdgeInfo e1 = pair.first; EdgeInfo e2 = pair.second;
        if (!check_edge_lookup(e1.side, e2.side)) return false;

        int p = e1.block_id; int ps = get_real_side(e1.side);
        int q = e2.block_id; int qs = get_real_side(e2.side);

       const EdgeInfo existing_p = S.adj[p][ps];
const EdgeInfo existing_q = S.adj[q][qs];

const bool p_used = existing_p.block_id != -1;
const bool q_used = existing_q.block_id != -1;

if (p_used || q_used) {
    // 只有双方已经精确互相配对，才属于重复拉链，可以跳过。
    const bool exact_existing_pair =
        p_used &&
        q_used &&
        existing_p.block_id == q &&
        get_real_side(existing_p.side) == qs &&
        existing_q.block_id == p &&
        get_real_side(existing_q.side) == ps;

    if (!exact_existing_pair) {
        return false;
    }

    continue;
}

        S.adj[p][ps] = { (int16_t)q, (int8_t)e2.side }; S.adj[q][qs] = { (int16_t)p, (int8_t)e1.side };
        bool same_dir = (S.block_type_cw[p] == S.block_type_cw[q]);
        int p_neighbors[2] = { get_prev_side(e1.side), get_next_side(e1.side) };
        int q_neighbors[2];
        if (same_dir) {
            q_neighbors[0] = get_next_side(e2.side); q_neighbors[1] = get_prev_side(e2.side);
        }
        else {
            q_neighbors[0] = get_prev_side(e2.side); q_neighbors[1] = get_next_side(e2.side);
        }

        for (int i = 0; i < 2; ++i) {
            int s_p_check = p_neighbors[i]; int s_q_check = q_neighbors[i];
            EdgeInfo link_p = S.adj[p][get_real_side(s_p_check)];
            EdgeInfo link_q = S.adj[q][get_real_side(s_q_check)];
            int k = -1;

            if (link_p.block_id != -1 && link_q.block_id != -1) {
                if (link_p.block_id != link_q.block_id) return false;
                k = link_p.block_id;
                int a1 = get_angle_val(s_p_check, e1.side); int a2 = get_angle_val(e2.side, s_q_check); int a3 = get_angle_val(link_p.side, link_q.side);
                if (!check_angle_lookup(a1, a2, a3)) return false;
            }
            else if (link_p.block_id != -1 && link_q.block_id == -1) {
                k = link_p.block_id; if (k == q) continue;
                bool same_dir_pk = (S.block_type_cw[p] == S.block_type_cw[k]); bool is_p_next = (s_p_check == get_next_side(e1.side));
                int s_k_target = same_dir_pk ? (is_p_next ? get_next_side(link_p.side) : get_prev_side(link_p.side)) : (is_p_next ? get_prev_side(link_p.side) : get_next_side(link_p.side));
                queue.push_back({ EdgeInfo{ (int16_t)q, (int8_t)s_q_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
            else if (link_q.block_id != -1 && link_p.block_id == -1) {
                k = link_q.block_id; if (k == p) continue;
                bool same_dir_qk = (S.block_type_cw[q] == S.block_type_cw[k]); bool is_q_next = (s_q_check == get_next_side(e2.side));
                int s_k_target = same_dir_qk ? (is_q_next ? get_next_side(link_q.side) : get_prev_side(link_q.side)) : (is_q_next ? get_prev_side(link_q.side) : get_next_side(link_q.side));
                queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)s_p_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
        }
    }
    return true;
}

// ==========================================================
// 严格一圈一圈铺设 + 预测剪枝(MRV) 候选选择器
// ==========================================================
struct Candidate {
    int p;
    int m;
    int score;
    bool is_dead_end;
    int num_valid_branches;
    int valid_branches[14];
};

Candidate get_best_candidate(const GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& temp_queue) {
    Candidate result = { -1, -1, -1, false, 0, {0} };

    int min_i = -1;
    for (int i = 0; i < S.block_count; ++i) {
        bool has_open = false;
        for (int j = 0; j < 7; ++j) {
            if (S.adj[i][j].block_id == -1) {
                has_open = true;
                break;
            }
        }
        if (has_open) {
            min_i = i;
            break;
        }
    }

    if (min_i == -1) return result;

    int max_score = -1;
    std::vector<int> candidates_j;
    candidates_j.reserve(7);

    for (int j = 0; j < 7; ++j) {
        if (S.adj[min_i][j].block_id == -1) {
            int score = 0;
            if (S.adj[min_i][get_real_side(get_prev_side(j))].block_id != -1) score += 10;
            if (S.adj[min_i][get_real_side(get_next_side(j))].block_id != -1) score += 10;

            if (score > max_score) {
                max_score = score;
                candidates_j.clear();
                candidates_j.push_back(j);
            }
            else if (score == max_score) {
                candidates_j.push_back(j);
            }
        }
    }

    int best_j = -1;
    int min_valid_branches = 999;
    int best_branches[14];
    int best_branches_count = 0;

    for (int j : candidates_j) {
        int valid_branches[14];
        int valid_count = 0;

        for (int n = 0; n < 14; ++n) {
            GlueState next_S = S;
            next_S.block_count++;
            int new_id = S.block_count;
            next_S.block_type_cw[new_id] = (n >= 7);
            int connect_side = (n < 7) ? n : n - 7;

            temp_queue.clear();
            temp_queue.push_back({ EdgeInfo{ (int16_t)min_i, (int8_t)j }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

            if (perform_zip_and_check(next_S, temp_queue)) {
                valid_branches[valid_count++] = n;
            }
        }

        if (valid_count == 0) {
            result.is_dead_end = true;
            return result;
        }

        if (valid_count < min_valid_branches) {
            min_valid_branches = valid_count;
            best_j = j;
            best_branches_count = valid_count;
            std::memcpy(best_branches, valid_branches, valid_count * sizeof(int));
        }
    }

    result.p = min_i;
    result.m = best_j;
    result.score = max_score;
    result.num_valid_branches = best_branches_count;
    std::memcpy(result.valid_branches, best_branches, best_branches_count * sizeof(int));

    return result;
}

void record_solution(const GlueState& S, int limit) {
    SolutionData sol;
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < 7; ++j) {
            sol.flat_data[i + j * limit] = (S.adj[i][j].block_id != -1) ? (S.adj[i][j].block_id + 1) * 100 + S.adj[i][j].side : 0;
        }
    }
#pragma omp critical(save_sol)
    {
        if (FIND_ALL_MODE || found_count.load(std::memory_order_relaxed) < TARGET_COUNT_LIMIT) {
            global_solutions.push_back(sol);
            if (!FIND_ALL_MODE && ++found_count >= TARGET_COUNT_LIMIT) {
                stop_search.store(true, std::memory_order_relaxed);
            }
        }
    }
}

// ==========================================================
// Phase 2: 动态任务窃取深度优先引擎
// ==========================================================
void worker_dfs_stealing(int depth, std::vector<GlueState>& path_states, std::vector<std::pair<EdgeInfo, EdgeInfo>>& zip_queue, int limit, uint32_t& rng_state) {
    if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

    const GlueState& S = path_states[depth];
    if (S.block_count == limit) { record_solution(S, limit); return; }

    Candidate cand = get_best_candidate(S, zip_queue);
    if (cand.is_dead_end || cand.p == -1) return;

    bool should_donate = false;
    if (depth < limit - 4 && idle_threads.load(std::memory_order_relaxed) > 0) {
        if (omp_test_lock(&pool_lock)) {
            if (global_task_pool.size() < 10000) { should_donate = true; }
            omp_unset_lock(&pool_lock);
        }
    }

    int p = cand.p; int m = cand.m; int new_id = S.block_count;
    int num_branches = cand.num_valid_branches;
    int branches[14];
    std::memcpy(branches, cand.valid_branches, num_branches * sizeof(int));

    if (!FIND_ALL_MODE) {
        for (int i = num_branches - 1; i > 0; --i) {
            int j = fast_rand(rng_state) % (i + 1); std::swap(branches[i], branches[j]);
        }
    }

    std::vector<GlueState> valid_children;
    if (should_donate) valid_children.reserve(num_branches);

    for (int i = 0; i < num_branches; ++i) {
        if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

        int n = branches[i];
        GlueState next_S = S;
        next_S.block_count++;
        next_S.block_type_cw[new_id] = (n >= 7);
        int connect_side = (n < 7) ? n : n - 7;

        zip_queue.clear();
        zip_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

        if (perform_zip_and_check(next_S, zip_queue)) {
            if (should_donate) {
                valid_children.push_back(next_S);
            }
            else {
                path_states[depth + 1] = next_S;
                worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
            }
        }
    }

    if (should_donate && !valid_children.empty()) {
        omp_set_lock(&pool_lock);
        for (size_t i = 1; i < valid_children.size(); ++i) {
            global_task_pool.push_back(valid_children[i]);
        }
        omp_unset_lock(&pool_lock);

        path_states[depth + 1] = valid_children[0];
        worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
    }
}

void build_encoded_lookup(const mxArray* mxData, int rows, int cols, std::vector<uint64_t>& table, bool is_edge) {
    double* pr = mxGetPr(mxData);
    table.clear(); table.reserve(rows);
    std::vector<int8_t> temp_row(cols);
    for (int r = 0; r < rows; ++r) {
        bool all_zero = true;
        for (int c = 0; c < cols; ++c) {
            temp_row[c] = (int8_t)std::round(pr[r + c * rows]);
            if (temp_row[c] != 0) all_zero = false;
        }
        if (all_zero) continue;
        if (is_edge) {
            int pos_idx = -1, neg_idx = -1;
            for (int c = 0; c < cols; ++c) {
                if (temp_row[c] == 1) pos_idx = c;
                if (temp_row[c] == -1) neg_idx = c;
            }
            if (pos_idx != -1 && neg_idx != -1) {
                std::fill(temp_row.begin(), temp_row.end(), 0);
                temp_row[std::min(pos_idx, neg_idx)] = 1; temp_row[std::max(pos_idx, neg_idx)] = -1;
            }
            table.push_back(encode_edge_vec(temp_row.data(), cols));
        }
        else {
            table.push_back(encode_angle_vec(temp_row.data(), cols));
        }
    }
    std::sort(table.begin(), table.end());
    table.erase(std::unique(table.begin(), table.end()), table.end());
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // 基础输入检查
    if (nrhs < 4) mexErrMsgTxt("Usage: heptagon_search(...)");

    int Target_N = (int)mxGetScalar(prhs[0]);
    if (Target_N > MAX_N) mexErrMsgIdAndTxt("Heptagon:TargetTooLarge", "Target_N exceeds MAX_N");

    const mxArray* mxE = prhs[1];
    build_encoded_lookup(mxE, (int)mxGetM(mxE), (int)mxGetN(mxE), LOOKUP_E_ENC, true);
    const mxArray* mxA = prhs[2];
    GLOBAL_COLS_A = (int)mxGetN(mxA);
    build_encoded_lookup(mxA, (int)mxGetM(mxA), GLOBAL_COLS_A, LOOKUP_A_ENC, false);

    FIND_ALL_MODE = (mxGetScalar(prhs[3]) > 0.5);
    TARGET_COUNT_LIMIT = (nrhs >= 5) ? (int)mxGetScalar(prhs[4]) : 1;

    // 获取第 6、7 个参数（若不传入，默认不跳过，值为 0）
    SKIP_EDGE_CHECK = (nrhs >= 6) ? (mxGetScalar(prhs[5]) > 0.5) : false;
    SKIP_ANGLE_CHECK = (nrhs >= 7) ? (mxGetScalar(prhs[6]) > 0.5) : false;

    if (SKIP_EDGE_CHECK)  mexPrintf(">> [System Notice] Edge checks are BYPASSED.\n");
    if (SKIP_ANGLE_CHECK) mexPrintf(">> [System Notice] Angle checks are BYPASSED.\n");

    global_solutions.clear();
    stop_search.store(false);
    found_count.store(0);
    global_task_pool.clear();
    idle_threads.store(0);
    omp_init_lock(&pool_lock);

    // =========================================================
    // Phase 1: 基础广度展开
    // =========================================================
    mexPrintf(">> [Phase 1] Adaptive Breadth Splitting...\n");

    GlueState initial_state;
    initial_state.block_count = 1;
    global_task_pool.push_back(initial_state);

    int current_depth = 1;
    while (global_task_pool.size() < MIN_FRAGMENTS && current_depth < Target_N) {
        if (stop_search.load(std::memory_order_relaxed)) break;
        std::vector<GlueState> next_pool;

#pragma omp parallel
        {
            std::vector<GlueState> local_next;
            local_next.reserve(1000);
            std::vector<std::pair<EdgeInfo, EdgeInfo>> local_queue;
            local_queue.reserve(64);

#pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < (int)global_task_pool.size(); ++i) {
                if (stop_search.load(std::memory_order_relaxed)) continue;
                const GlueState& S = global_task_pool[i];

                Candidate cand = get_best_candidate(S, local_queue);
                if (cand.is_dead_end || cand.p == -1) continue;

                int p = cand.p; int m = cand.m; int new_id = S.block_count;
                for (int b = 0; b < cand.num_valid_branches; ++b) {
                    int n = cand.valid_branches[b];
                    GlueState next_S = S;
                    next_S.block_count++;
                    next_S.block_type_cw[new_id] = (n >= 7);
                    int connect_side = (n < 7) ? n : n - 7;
                    local_queue.clear();
                    local_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

                    if (perform_zip_and_check(next_S, local_queue)) {
                        if (next_S.block_count == Target_N) record_solution(next_S, Target_N);
                        else local_next.push_back(next_S);
                    }
                }
            }
#pragma omp critical(merge_pool)
            { next_pool.insert(next_pool.end(), local_next.begin(), local_next.end()); }
        }
        global_task_pool = std::move(next_pool);
        current_depth++;
        if (global_task_pool.size() > MAX_FRAGMENTS) break;
    }

    if (!FIND_ALL_MODE && !global_task_pool.empty()) {
        srand((unsigned int)time(NULL));
        std::random_shuffle(global_task_pool.begin(), global_task_pool.end());
    }
    mexPrintf(">> Generated %d seeds. [Phase 2] Dynamic Work-Stealing Activated!\n", global_task_pool.size());

    // =========================================================
    // Phase 2: 动态多核窃取引擎
    // =========================================================
#pragma omp parallel
    {
#pragma omp single
        { num_total_threads = omp_get_num_threads(); }

        std::vector<GlueState> thread_path(Target_N + 5);
        std::vector<std::pair<EdgeInfo, EdgeInfo>> thread_zip_queue;
        thread_zip_queue.reserve(64);
        uint32_t thread_rng = 123456789 ^ omp_get_thread_num() ^ (unsigned int)time(NULL);

        while (!stop_search.load(std::memory_order_relaxed)) {
            GlueState task;
            bool got_task = false;

            omp_set_lock(&pool_lock);
            if (!global_task_pool.empty()) {
                task = global_task_pool.back();
                global_task_pool.pop_back();
                got_task = true;
            }
            omp_unset_lock(&pool_lock);

            if (got_task) {
                thread_path[task.block_count] = task;
                worker_dfs_stealing(task.block_count, thread_path, thread_zip_queue, Target_N, thread_rng);
            }
            else {
                idle_threads.fetch_add(1, std::memory_order_relaxed);

                while (global_task_pool.empty() && !stop_search.load(std::memory_order_relaxed)) {
                    if (idle_threads.load(std::memory_order_relaxed) == num_total_threads) {
                        stop_search.store(true, std::memory_order_relaxed);
                        break;
                    }
                    std::this_thread::yield();
                }

                idle_threads.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    omp_destroy_lock(&pool_lock);

    // =========================================================
    // 返回数据
    // =========================================================
    int count = (int)global_solutions.size();
    const char* fields[] = { "count", "solutions" };
    plhs[0] = mxCreateStructMatrix(1, 1, 2, fields);
    mxSetField(plhs[0], 0, "count", mxCreateDoubleScalar((double)count));

    mxArray* out_cell = mxCreateCellMatrix(1, count);
    for (int i = 0; i < count; ++i) {
        mxArray* m = mxCreateNumericMatrix(Target_N, 7, mxINT32_CLASS, mxREAL);
        int* d = (int*)mxGetData(m);
        for (int k = 0; k < Target_N * 7; ++k) d[k] = global_solutions[i].flat_data[k];
        mxSetCell(out_cell, i, m);
    }
    mxSetField(plhs[0], 0, "solutions", out_cell);
}



// 优化版多核调度 - 支持手动短路跳过边/角验证 + 2阶顶点子集预判剪枝优化 (去通信死锁稳定版)
/*#include "mex.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <thread>
#include <random>
#include <ctime> 

// ==========================================================
// 基础配置与全局开关
// ==========================================================
const int MAX_N = 1000;

bool SKIP_EDGE_CHECK = false;   // 手动控制开关：跳过边验证
bool SKIP_ANGLE_CHECK = false;  // 手动控制开关：跳过角验证

inline int get_real_side(int s) { return s % 7; }
inline int get_prev_side(int s) { return (s < 7) ? (s + 6) % 7 : 7 + (s - 7 + 6) % 7; }
inline int get_next_side(int s) { return (s < 7) ? (s + 1) % 7 : 7 + (s - 7 + 1) % 7; }
inline int get_angle_val(int s1, int s2) {
    int r1 = s1 % 7; int r2 = s2 % 7;
    if ((r1 == 0 && r2 == 6) || (r1 == 6 && r2 == 0)) return 0;
    return std::max(r1, r2);
}

std::vector<uint64_t> LOOKUP_E_ENC;
std::vector<uint64_t> LOOKUP_A_ENC;
int GLOBAL_COLS_A = 8;

// ==========================================================
// 【优化】超快 O(1) 全局查找表，免去运行时二分查找与编码开销
// ==========================================================
bool VALID_2_DEGREE[8][8] = { false };
bool VALID_3_DEGREE[8][8][8] = { false };
bool VALID_EDGE_TRANSITION[7][7] = { false };

struct EdgeInfo {
    int16_t block_id;
    int8_t side;
};

struct GlueState {
    EdgeInfo adj[MAX_N][7];
    int16_t block_count;
    bool block_type_cw[MAX_N];

    GlueState() {
        block_count = 0;
        std::memset(block_type_cw, 0, MAX_N * sizeof(bool));
        for (int i = 0; i < MAX_N; ++i) {
            for (int j = 0; j < 7; ++j) {
                adj[i][j].block_id = -1;
                adj[i][j].side = -1;
            }
        }
    }
};

// 增量状态复制
inline void copy_state(GlueState& dest, const GlueState& src) {
    dest.block_count = src.block_count;
    std::memcpy(dest.adj, src.adj, src.block_count * sizeof(src.adj[0]));
    std::memcpy(dest.block_type_cw, src.block_type_cw, src.block_count * sizeof(bool));

    for (int j = 0; j < 7; ++j) {
        dest.adj[src.block_count][j].block_id = -1;
        dest.adj[src.block_count][j].side = -1;
    }
}

inline uint64_t encode_edge_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = code * 3 + (row[i] + 1);
    return code;
}
inline uint64_t encode_angle_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = (code << 3) | (row[i] & 0x7);
    return code;
}

// 初始化全局高速过滤表
void init_all_lookups(const mxArray* mxA, int cols_A) {
    int rows_A = (int)mxGetM(mxA);
    double* pr_A = mxGetPr(mxA);

    std::memset(VALID_2_DEGREE, 0, sizeof(VALID_2_DEGREE));
    for (int r = 0; r < rows_A; ++r) {
        int counts[7] = { 0 };
        for (int c = 0; c < 7 && c < cols_A - 1; ++c) {
            counts[c] = (int)std::round(pr_A[r + c * rows_A]);
        }
        std::vector<int> angles;
        for (int i = 0; i < 7; ++i) {
            for (int k = 0; k < counts[i]; ++k) {
                angles.push_back(i);
            }
        }
        if (angles.size() == 3) {
            int a = angles[0]; int b = angles[1]; int c = angles[2];
            if (a < 8 && b < 8 && c < 8) {
                VALID_2_DEGREE[a][b] = true; VALID_2_DEGREE[b][a] = true;
                VALID_2_DEGREE[b][c] = true; VALID_2_DEGREE[c][b] = true;
                VALID_2_DEGREE[a][c] = true; VALID_2_DEGREE[c][a] = true;
            }
        }
    }

    std::memset(VALID_3_DEGREE, 0, sizeof(VALID_3_DEGREE));
    for (int a1 = 0; a1 < 8; ++a1) {
        for (int a2 = 0; a2 < 8; ++a2) {
            for (int a3 = 0; a3 < 8; ++a3) {
                uint64_t code = 0; int8_t temp[32] = { 0 };
                if (a1 < cols_A - 1) temp[a1]++;
                if (a2 < cols_A - 1) temp[a2]++;
                if (a3 < cols_A - 1) temp[a3]++;
                if (cols_A >= 8) temp[7] = 2;
                for (int i = 0; i < cols_A; ++i) { code = (code << 3) | (temp[i] & 0x7); }
                auto it = std::lower_bound(LOOKUP_A_ENC.begin(), LOOKUP_A_ENC.end(), code);
                if (it != LOOKUP_A_ENC.end() && *it == code) {
                    VALID_3_DEGREE[a1][a2][a3] = true;
                }
            }
        }
    }

    std::memset(VALID_EDGE_TRANSITION, 0, sizeof(VALID_EDGE_TRANSITION));
    for (int u = 0; u < 7; ++u) {
        for (int v = 0; v < 7; ++v) {
            if (u == v) {
                VALID_EDGE_TRANSITION[u][v] = true;
            }
            else {
                int min_s = std::min(u, v); int max_s = std::max(u, v);
                uint64_t code = 0;
                for (int i = 0; i < 7; ++i) {
                    int val = 1;
                    if (i == min_s) val = 2; else if (i == max_s) val = 0;
                    code = code * 3 + val;
                }
                auto it = std::lower_bound(LOOKUP_E_ENC.begin(), LOOKUP_E_ENC.end(), code);
                VALID_EDGE_TRANSITION[u][v] = (it != LOOKUP_E_ENC.end() && *it == code);
            }
        }
    }
}

struct SolutionData {
    int flat_data[MAX_N * 7];
    SolutionData() { std::memset(flat_data, 0, sizeof(flat_data)); }
};

// ==========================================================
// 全局状态与动态负载均衡控制
// ==========================================================
std::vector<SolutionData> global_solutions;
std::atomic<bool> stop_search(false);
std::atomic<int> found_count(0);
bool FIND_ALL_MODE = false;
int TARGET_COUNT_LIMIT = 1;

std::vector<GlueState> global_task_pool;
omp_lock_t pool_lock;
std::atomic<int> idle_threads(0);
int num_total_threads = 1;

inline uint32_t fast_rand(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

inline bool check_edge_lookup(int side_p, int side_q) {
    if (SKIP_EDGE_CHECK) return true;
    return VALID_EDGE_TRANSITION[get_real_side(side_p)][get_real_side(side_q)];
}

inline bool check_angle_lookup(int a1, int a2, int a3) {
    if (SKIP_ANGLE_CHECK) return true;
    if (a1 >= 8 || a2 >= 8 || a3 >= 8 || a1 < 0 || a2 < 0 || a3 < 0) return false;
    return VALID_3_DEGREE[a1][a2][a3];
}

// 拓扑传导与闭合检查
bool perform_zip_and_check(GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& queue) {
    size_t head = 0;
    while (head < queue.size()) {
        auto pair = queue[head++];
        EdgeInfo e1 = pair.first; EdgeInfo e2 = pair.second;
        if (!check_edge_lookup(e1.side, e2.side)) return false;

        int p = e1.block_id; int ps = get_real_side(e1.side);
        int q = e2.block_id; int qs = get_real_side(e2.side);

        const EdgeInfo existing_p = S.adj[p][ps];
        const EdgeInfo existing_q = S.adj[q][qs];
        const bool p_used = existing_p.block_id != -1;
        const bool q_used = existing_q.block_id != -1;

        if (p_used || q_used) {
            // An already-used edge is valid only when both adjacency records
            // describe this exact reciprocal edge pairing.  Reaching the same
            // pairing through another zipper path is an idempotent no-op.
            const bool exact_existing_pair =
                p_used && q_used &&
                existing_p.block_id == q &&
                get_real_side(existing_p.side) == qs &&
                existing_q.block_id == p &&
                get_real_side(existing_q.side) == ps;

            if (!exact_existing_pair) {
                return false;
            }
            continue;
        }

        S.adj[p][ps] = { (int16_t)q, (int8_t)e2.side }; S.adj[q][qs] = { (int16_t)p, (int8_t)e1.side };
        bool same_dir = (S.block_type_cw[p] == S.block_type_cw[q]);
        int p_neighbors[2] = { get_prev_side(e1.side), get_next_side(e1.side) };
        int q_neighbors[2];
        if (same_dir) {
            q_neighbors[0] = get_next_side(e2.side); q_neighbors[1] = get_prev_side(e2.side);
        }
        else {
            q_neighbors[0] = get_prev_side(e2.side); q_neighbors[1] = get_next_side(e2.side);
        }

        for (int i = 0; i < 2; ++i) {
            int s_p_check = p_neighbors[i]; int s_q_check = q_neighbors[i];

            if (!SKIP_ANGLE_CHECK) {
                int a1 = get_angle_val(s_p_check, e1.side);
                int a2 = get_angle_val(e2.side, s_q_check);
                if (a1 < 8 && a2 < 8 && !VALID_2_DEGREE[a1][a2]) {
                    return false;
                }
            }

            EdgeInfo link_p = S.adj[p][get_real_side(s_p_check)];
            EdgeInfo link_q = S.adj[q][get_real_side(s_q_check)];
            int k = -1;

            if (link_p.block_id != -1 && link_q.block_id != -1) {
                if (link_p.block_id != link_q.block_id) return false;
                k = link_p.block_id;
                int a1 = get_angle_val(s_p_check, e1.side); int a2 = get_angle_val(e2.side, s_q_check); int a3 = get_angle_val(link_p.side, link_q.side);
                if (!check_angle_lookup(a1, a2, a3)) return false;
            }
            else if (link_p.block_id != -1 && link_q.block_id == -1) {
                k = link_p.block_id; if (k == q) continue;
                bool same_dir_pk = (S.block_type_cw[p] == S.block_type_cw[k]); bool is_p_next = (s_p_check == get_next_side(e1.side));
                int s_k_target = same_dir_pk ? (is_p_next ? get_next_side(link_p.side) : get_prev_side(link_p.side)) : (is_p_next ? get_prev_side(link_p.side) : get_next_side(link_p.side));
                queue.push_back({ EdgeInfo{ (int16_t)q, (int8_t)s_q_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
            else if (link_q.block_id != -1 && link_p.block_id == -1) {
                k = link_q.block_id; if (k == p) continue;
                bool same_dir_qk = (S.block_type_cw[q] == S.block_type_cw[k]); bool is_q_next = (s_q_check == get_next_side(e2.side));
                int s_k_target = same_dir_qk ? (is_q_next ? get_next_side(link_q.side) : get_prev_side(link_q.side)) : (is_q_next ? get_prev_side(link_q.side) : get_next_side(link_q.side));
                queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)s_p_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
        }
    }
    return true;
}

// MRV 候选选择器
struct Candidate {
    int p;
    int m;
    int score;
    bool is_dead_end;
    int num_valid_branches;
    int valid_branches[14];
};

Candidate get_best_candidate(const GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& temp_queue) {
    Candidate result = { -1, -1, -1, false, 0, {0} };

    int min_i = -1;
    for (int i = 0; i < S.block_count; ++i) {
        bool has_open = false;
        for (int j = 0; j < 7; ++j) {
            if (S.adj[i][j].block_id == -1) {
                has_open = true;
                break;
            }
        }
        if (has_open) {
            min_i = i;
            break;
        }
    }

    if (min_i == -1) return result;

    int max_score = -1;
    std::vector<int> candidates_j;
    candidates_j.reserve(7);

    for (int j = 0; j < 7; ++j) {
        if (S.adj[min_i][j].block_id == -1) {
            int score = 0;
            if (S.adj[min_i][get_real_side(get_prev_side(j))].block_id != -1) score += 10;
            if (S.adj[min_i][get_real_side(get_next_side(j))].block_id != -1) score += 10;

            if (score > max_score) {
                max_score = score;
                candidates_j.clear();
                candidates_j.push_back(j);
            }
            else if (score == max_score) {
                candidates_j.push_back(j);
            }
        }
    }

    int best_j = -1;
    int min_valid_branches = 999;
    int best_branches[14];
    int best_branches_count = 0;

    GlueState next_S;

    for (int j : candidates_j) {
        int valid_branches[14];
        int valid_count = 0;

        for (int n = 0; n < 14; ++n) {
            copy_state(next_S, S);
            next_S.block_count++;
            int new_id = S.block_count;
            next_S.block_type_cw[new_id] = (n >= 7);
            int connect_side = (n < 7) ? n : n - 7;

            temp_queue.clear();
            temp_queue.push_back({ EdgeInfo{ (int16_t)min_i, (int8_t)j }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

            if (perform_zip_and_check(next_S, temp_queue)) {
                valid_branches[valid_count++] = n;
            }
        }

        if (valid_count == 0) {
            result.is_dead_end = true;
            return result;
        }

        if (valid_count < min_valid_branches) {
            min_valid_branches = valid_count;
            best_j = j;
            best_branches_count = valid_count;
            std::memcpy(best_branches, valid_branches, valid_count * sizeof(int));
        }
    }

    result.p = min_i;
    result.m = best_j;
    result.score = max_score;
    result.num_valid_branches = best_branches_count;
    std::memcpy(result.valid_branches, best_branches, best_branches_count * sizeof(int));

    return result;
}

void record_solution(const GlueState& S, int limit) {
    SolutionData sol;
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < 7; ++j) {
            if (S.adj[i][j].block_id == -1) {
                sol.flat_data[i + j * limit] = 0;
                continue;
            }

            // Side codes 0..6 mean CCW; 7..13 mean CW.
            // Preserve block_type_cw in the encoded result.
            const int neighbor_id = S.adj[i][j].block_id;
            const int encoded_side = S.adj[i][j].side
                + (S.block_type_cw[neighbor_id] ? 7 : 0);
            sol.flat_data[i + j * limit] = (neighbor_id + 1) * 100 + encoded_side;
        }
    }
#pragma omp critical(save_sol)
    {
        if (FIND_ALL_MODE || found_count.load(std::memory_order_relaxed) < TARGET_COUNT_LIMIT) {
            global_solutions.push_back(sol);
            if (!FIND_ALL_MODE && ++found_count >= TARGET_COUNT_LIMIT) {
                stop_search.store(true, std::memory_order_relaxed);
            }
        }
    }
}

// Phase 2: 动态任务窃取深度优先引擎 (去交互死锁版)
void worker_dfs_stealing(int depth, std::vector<GlueState>& path_states, std::vector<std::pair<EdgeInfo, EdgeInfo>>& zip_queue, int limit, uint32_t& rng_state) {
    if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

    const GlueState& S = path_states[depth];
    if (S.block_count == limit) { record_solution(S, limit); return; }

    Candidate cand = get_best_candidate(S, zip_queue);
    if (cand.is_dead_end || cand.p == -1) return;

    bool should_donate = false;
    if (depth < limit - 4 && idle_threads.load(std::memory_order_relaxed) > 0) {
        if (omp_test_lock(&pool_lock)) {
            if (global_task_pool.size() < 10000) { should_donate = true; }
            omp_unset_lock(&pool_lock);
        }
    }

    int p = cand.p; int m = cand.m; int new_id = S.block_count;
    int num_branches = cand.num_valid_branches;
    int branches[14];
    std::memcpy(branches, cand.valid_branches, num_branches * sizeof(int));

    if (!FIND_ALL_MODE) {
        for (int i = num_branches - 1; i > 0; --i) {
            int j = fast_rand(rng_state) % (i + 1); std::swap(branches[i], branches[j]);
        }
    }

    std::vector<GlueState> valid_children;
    if (should_donate) valid_children.reserve(num_branches);

    for (int i = 0; i < num_branches; ++i) {
        if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

        int n = branches[i];

        copy_state(path_states[depth + 1], S);
        GlueState& next_S = path_states[depth + 1];
        next_S.block_count++;
        next_S.block_type_cw[new_id] = (n >= 7);
        int connect_side = (n < 7) ? n : n - 7;

        zip_queue.clear();
        zip_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

        if (perform_zip_and_check(next_S, zip_queue)) {
            if (should_donate) {
                valid_children.push_back(next_S);
            }
            else {
                worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
            }
        }
    }

    if (should_donate && !valid_children.empty()) {
        omp_set_lock(&pool_lock);
        for (size_t i = 1; i < valid_children.size(); ++i) {
            global_task_pool.push_back(valid_children[i]);
        }
        omp_unset_lock(&pool_lock);

        copy_state(path_states[depth + 1], valid_children[0]);
        worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
    }
}

void build_encoded_lookup(const mxArray* mxData, int rows, int cols, std::vector<uint64_t>& table, bool is_edge) {
    double* pr = mxGetPr(mxData);
    table.clear(); table.reserve(rows);
    std::vector<int8_t> temp_row(cols);
    for (int r = 0; r < rows; ++r) {
        bool all_zero = true;
        for (int c = 0; c < cols; ++c) {
            temp_row[c] = (int8_t)std::round(pr[r + c * rows]);
            if (temp_row[c] != 0) all_zero = false;
        }
        if (all_zero) continue;
        if (is_edge) {
            int pos_idx = -1, neg_idx = -1;
            for (int c = 0; c < cols; ++c) {
                if (temp_row[c] == 1) pos_idx = c;
                if (temp_row[c] == -1) neg_idx = c;
            }
            if (pos_idx != -1 && neg_idx != -1) {
                std::fill(temp_row.begin(), temp_row.end(), 0);
                temp_row[std::min(pos_idx, neg_idx)] = 1; temp_row[std::max(pos_idx, neg_idx)] = -1;
            }
            table.push_back(encode_edge_vec(temp_row.data(), cols));
        }
        else {
            table.push_back(encode_angle_vec(temp_row.data(), cols));
        }
    }
    std::sort(table.begin(), table.end());
    table.erase(std::unique(table.begin(), table.end()), table.end());
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 4) mexErrMsgTxt("Usage: heptagon_search(...)");

    int Target_N = (int)mxGetScalar(prhs[0]);
    if (Target_N > MAX_N) mexErrMsgIdAndTxt("Heptagon:TargetTooLarge", "Target_N exceeds MAX_N");

    const mxArray* mxE = prhs[1];
    build_encoded_lookup(mxE, (int)mxGetM(mxE), (int)mxGetN(mxE), LOOKUP_E_ENC, true);
    const mxArray* mxA = prhs[2];
    GLOBAL_COLS_A = (int)mxGetN(mxA);
    build_encoded_lookup(mxA, (int)mxGetM(mxA), GLOBAL_COLS_A, LOOKUP_A_ENC, false);

    init_all_lookups(mxA, GLOBAL_COLS_A);

    FIND_ALL_MODE = (mxGetScalar(prhs[3]) > 0.5);
    TARGET_COUNT_LIMIT = (nrhs >= 5) ? (int)mxGetScalar(prhs[4]) : 1;

    SKIP_EDGE_CHECK = (nrhs >= 6) ? (mxGetScalar(prhs[5]) > 0.5) : false;
    SKIP_ANGLE_CHECK = (nrhs >= 7) ? (mxGetScalar(prhs[6]) > 0.5) : false;

    if (SKIP_EDGE_CHECK)  mexPrintf(">> [System Notice] Edge checks are BYPASSED.\n");
    if (SKIP_ANGLE_CHECK) mexPrintf(">> [System Notice] Angle checks are BYPASSED.\n");

    global_solutions.clear();
    stop_search.store(false);
    found_count.store(0);
    global_task_pool.clear();
    idle_threads.store(0);
    omp_init_lock(&pool_lock);

    // =========================================================
    // 【硬件自适应】根据当前 CPU 的物理线程数动态计算最佳种子数量
    // =========================================================
    int num_threads = std::thread::hardware_concurrency(); // 获取系统支持的并发线程数
    if (num_threads < 1) num_threads = 1;

    size_t min_fragments;
    size_t max_fragments;

    if (FIND_ALL_MODE) {
        // 找所有解模式：每个核心至少分配 50 个种子任务，保障深度负载均衡，最低 2000
        min_fragments = std::max((size_t)2000, (size_t)num_threads * 50);
        max_fragments = min_fragments * 6;
    }
    else {
        // 找单个解模式：每个核心分配 8 个种子作为缓冲，既保证多核饱和，又避免过度展开，最低 120
        min_fragments = std::max((size_t)120, (size_t)num_threads * 8);
        max_fragments = min_fragments * 4;
    }

    mexPrintf(">> [Phase 1] Hardware-Adaptive Splitting (Threads: %d, Target Seeds: %d)...\n",
        num_threads, (int)min_fragments);

    GlueState initial_state;
    initial_state.block_count = 1;
    global_task_pool.push_back(initial_state);

    int current_depth = 1;
    while (global_task_pool.size() < min_fragments && current_depth < Target_N) {
        if (stop_search.load(std::memory_order_relaxed)) break;
        std::vector<GlueState> next_pool;

#pragma omp parallel
        {
            std::vector<GlueState> local_next;
            local_next.reserve(1000);
            std::vector<std::pair<EdgeInfo, EdgeInfo>> local_queue;
            local_queue.reserve(64);

#pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < (int)global_task_pool.size(); ++i) {
                if (stop_search.load(std::memory_order_relaxed)) continue;
                const GlueState& S = global_task_pool[i];

                Candidate cand = get_best_candidate(S, local_queue);
                if (cand.is_dead_end || cand.p == -1) continue;

                int p = cand.p; int m = cand.m; int new_id = S.block_count;
                for (int b = 0; b < cand.num_valid_branches; ++b) {
                    int n = cand.valid_branches[b];

                    GlueState next_S;
                    copy_state(next_S, S);
                    next_S.block_count++;
                    next_S.block_type_cw[new_id] = (n >= 7);
                    int connect_side = (n < 7) ? n : n - 7;
                    local_queue.clear();
                    local_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

                    if (perform_zip_and_check(next_S, local_queue)) {
                        if (next_S.block_count == Target_N) record_solution(next_S, Target_N);
                        else local_next.push_back(next_S);
                    }
                }
            }
#pragma omp critical(merge_pool)
            { next_pool.insert(next_pool.end(), local_next.begin(), local_next.end()); }
        }
        global_task_pool = std::move(next_pool);
        current_depth++;
        if (global_task_pool.size() > max_fragments) break;
    }

    if (!FIND_ALL_MODE && !global_task_pool.empty()) {
        std::mt19937 g(static_cast<unsigned int>(time(NULL)));
        std::shuffle(global_task_pool.begin(), global_task_pool.end(), g);
    }

    mexPrintf(">> Generated %d seeds. [Phase 2] Dynamic Work-Stealing Activated!\n", (int)global_task_pool.size());

    // =========================================================
    // Phase 2: 动态多核窃取引擎
    // =========================================================
#pragma omp parallel
    {
#pragma omp single
        { num_total_threads = omp_get_num_threads(); }

        std::vector<GlueState> thread_path(Target_N + 5);
        std::vector<std::pair<EdgeInfo, EdgeInfo>> thread_zip_queue;
        thread_zip_queue.reserve(64);
        uint32_t thread_rng = 123456789 ^ omp_get_thread_num() ^ (unsigned int)time(NULL);

        while (!stop_search.load(std::memory_order_relaxed)) {
            GlueState task;
            bool got_task = false;

            omp_set_lock(&pool_lock);
            if (!global_task_pool.empty()) {
                task = global_task_pool.back();
                global_task_pool.pop_back();
                got_task = true;
            }
            omp_unset_lock(&pool_lock);

            if (got_task) {
                copy_state(thread_path[task.block_count], task);
                worker_dfs_stealing(task.block_count, thread_path, thread_zip_queue, Target_N, thread_rng);
            }
            else {
                idle_threads.fetch_add(1, std::memory_order_relaxed);

                while (global_task_pool.empty() && !stop_search.load(std::memory_order_relaxed)) {
                    if (idle_threads.load(std::memory_order_relaxed) == num_total_threads) {
                        stop_search.store(true, std::memory_order_relaxed);
                        break;
                    }
                    std::this_thread::yield();
                }

                idle_threads.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    omp_destroy_lock(&pool_lock);

    // =========================================================
    // 返回数据
    // =========================================================
    int count = (int)global_solutions.size();
    const char* fields[] = { "count", "solutions" };
    plhs[0] = mxCreateStructMatrix(1, 1, 2, fields);
    mxSetField(plhs[0], 0, "count", mxCreateDoubleScalar((double)count));

    mxArray* out_cell = mxCreateCellMatrix(1, count);
    for (int i = 0; i < count; ++i) {
        mxArray* m = mxCreateNumericMatrix(Target_N, 7, mxINT32_CLASS, mxREAL);
        int* d = (int*)mxGetData(m);
        for (int k = 0; k < Target_N * 7; ++k) d[k] = global_solutions[i].flat_data[k];
        mxSetCell(out_cell, i, m);
    }
    mxSetField(plhs[0], 0, "solutions", out_cell);
}








// 终极高性能多核调度版 - 动态任务窃取(无Task防崩) + 零排队 + 零爆栈 + 超时安全强制中断+时间控制！！
/*#include "mex.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <thread>
#include <chrono> // 引入高精度时间库

const int MAX_N = 50;
const size_t MIN_FRAGMENTS = 3000;
const size_t MAX_FRAGMENTS = 20000;

inline int get_real_side(int s) { return s % 7; }
inline int get_prev_side(int s) { return (s < 7) ? (s + 6) % 7 : 7 + (s - 7 + 6) % 7; }
inline int get_next_side(int s) { return (s < 7) ? (s + 1) % 7 : 7 + (s - 7 + 1) % 7; }
inline int get_angle_val(int s1, int s2) {
    int r1 = s1 % 7; int r2 = s2 % 7;
    if ((r1 == 0 && r2 == 6) || (r1 == 6 && r2 == 0)) return 0;
    return std::max(r1, r2);
}

std::vector<uint64_t> LOOKUP_E_ENC;
std::vector<uint64_t> LOOKUP_A_ENC;
int GLOBAL_COLS_A = 8;

// 静态查找表
static bool ALLOWED_E[7][7];
static bool ALLOWED_A[8][8][8];

// 超时控制全局参数
std::chrono::steady_clock::time_point global_start_time;
double MAX_DURATION_SECONDS = 600.0;
std::atomic<bool> is_timeout(false);

inline uint64_t encode_edge_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = code * 3 + (row[i] + 1);
    return code;
}
inline uint64_t encode_angle_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = (code << 3) | (row[i] & 0x7);
    return code;
}

struct EdgeInfo {
    int16_t block_id;
    int8_t side;
};

struct GlueState {
    EdgeInfo adj[MAX_N][7];
    int16_t block_count;
    bool block_type_cw[MAX_N];

    GlueState() {
        block_count = 0;
        std::memset(block_type_cw, 0, MAX_N * sizeof(bool));
        for (int i = 0; i < MAX_N; ++i) {
            for (int j = 0; j < 7; ++j) {
                adj[i][j].block_id = -1;
                adj[i][j].side = -1;
            }
        }
    }
};

struct SolutionData {
    int flat_data[MAX_N * 7];
    SolutionData() { std::memset(flat_data, 0, sizeof(flat_data)); }
};

std::vector<SolutionData> global_solutions;
std::atomic<bool> stop_search(false);
std::atomic<int> found_count(0);
bool FIND_ALL_MODE = false;
int TARGET_COUNT_LIMIT = 1;

std::vector<GlueState> global_task_pool;
omp_lock_t pool_lock;
std::atomic<int> idle_threads(0);
int num_total_threads = 1;

inline uint32_t fast_rand(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

inline bool check_edge_lookup(int side_p, int side_q) {
    return ALLOWED_E[get_real_side(side_p)][get_real_side(side_q)];
}

inline bool check_angle_lookup(int a1, int a2, int a3) {
    if (a1 < 0 || a1 > 7 || a2 < 0 || a2 > 7 || a3 < 0 || a3 > 7) return false;
    return ALLOWED_A[a1][a2][a3];
}

bool perform_zip_and_check(GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& queue) {
    size_t head = 0;
    while (head < queue.size()) {
        auto pair = queue[head++];
        EdgeInfo e1 = pair.first; EdgeInfo e2 = pair.second;
        if (!check_edge_lookup(e1.side, e2.side)) return false;

        int p = e1.block_id; int ps = get_real_side(e1.side);
        int q = e2.block_id; int qs = get_real_side(e2.side);

        if (S.adj[p][ps].block_id != -1) {
            if (S.adj[p][ps].block_id != q) return false;
            continue;
        }
        if (S.adj[q][qs].block_id != -1) {
            if (S.adj[q][qs].block_id != p) return false;
            continue;
        }

        S.adj[p][ps] = { (int16_t)q, (int8_t)e2.side }; S.adj[q][qs] = { (int16_t)p, (int8_t)e1.side };
        bool same_dir = (S.block_type_cw[p] == S.block_type_cw[q]);
        int p_neighbors[2] = { get_prev_side(e1.side), get_next_side(e1.side) };
        int q_neighbors[2];
        if (same_dir) {
            q_neighbors[0] = get_next_side(e2.side); q_neighbors[1] = get_prev_side(e2.side);
        }
        else {
            q_neighbors[0] = get_prev_side(e2.side); q_neighbors[1] = get_next_side(e2.side);
        }

        for (int i = 0; i < 2; ++i) {
            int s_p_check = p_neighbors[i]; int s_q_check = q_neighbors[i];
            EdgeInfo link_p = S.adj[p][get_real_side(s_p_check)];
            EdgeInfo link_q = S.adj[q][get_real_side(s_q_check)];
            int k = -1;

            if (link_p.block_id != -1 && link_q.block_id != -1) {
                if (link_p.block_id != link_q.block_id) return false;
                k = link_p.block_id;
                int a1 = get_angle_val(s_p_check, e1.side); int a2 = get_angle_val(e2.side, s_q_check); int a3 = get_angle_val(link_p.side, link_q.side);
                if (!check_angle_lookup(a1, a2, a3)) return false;
            }
            else if (link_p.block_id != -1 && link_q.block_id == -1) {
                k = link_p.block_id; if (k == q) continue;
                bool same_dir_pk = (S.block_type_cw[p] == S.block_type_cw[k]); bool is_p_next = (s_p_check == get_next_side(e1.side));
                int s_k_target = same_dir_pk ? (is_p_next ? get_next_side(link_p.side) : get_prev_side(link_p.side)) : (is_p_next ? get_prev_side(link_p.side) : get_next_side(link_p.side));
                queue.push_back({ EdgeInfo{ (int16_t)q, (int8_t)s_q_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
            else if (link_q.block_id != -1 && link_p.block_id == -1) {
                k = link_q.block_id; if (k == p) continue;
                bool same_dir_qk = (S.block_type_cw[q] == S.block_type_cw[k]); bool is_q_next = (s_q_check == get_next_side(e2.side));
                int s_k_target = same_dir_qk ? (is_q_next ? get_next_side(link_q.side) : get_prev_side(link_q.side)) : (is_q_next ? get_prev_side(link_q.side) : get_next_side(link_q.side));
                queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)s_p_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
        }
    }
    return true;
}

struct Candidate { int p; int m; int score; };
Candidate get_best_candidate(const GlueState& S) {
    Candidate best = { -1, -1, -1 };
    for (int i = 0; i < S.block_count; ++i) {
        for (int j = 0; j < 7; ++j) {
            if (S.adj[i][j].block_id == -1) {
                int score = 0;
                if (S.adj[i][get_real_side(get_prev_side(j))].block_id != -1) score += 10;
                if (S.adj[i][get_real_side(get_next_side(j))].block_id != -1) score += 10;
                if (score > best.score) best = { i, j, score };
            }
        }
    }
    return best;
}

void record_solution(const GlueState& S, int limit) {
    SolutionData sol;
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < 7; ++j) {
            sol.flat_data[i + j * limit] = (S.adj[i][j].block_id != -1) ? (S.adj[i][j].block_id + 1) * 100 + S.adj[i][j].side : 0;
        }
    }
#pragma omp critical(save_sol)
    {
        if (FIND_ALL_MODE || found_count.load(std::memory_order_relaxed) < TARGET_COUNT_LIMIT) {
            global_solutions.push_back(sol);
            if (!FIND_ALL_MODE && ++found_count >= TARGET_COUNT_LIMIT) {
                stop_search.store(true, std::memory_order_relaxed);
            }
        }
    }
}

// ==========================================================
// Phase 2: DFS 引擎 (具有低开销高精度的定期超时检测)
// ==========================================================
void worker_dfs_stealing(int depth, std::vector<GlueState>& path_states, std::vector<std::pair<EdgeInfo, EdgeInfo>>& zip_queue, int limit, uint32_t& rng_state) {
    if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

    // 【低开销超时检测】：位与计算（2的10次方）可以避免在每个递归周期里高频请求系统时间
    if ((rng_state & 0x3FF) == 0) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - global_start_time).count();
        if (elapsed > MAX_DURATION_SECONDS) {
            is_timeout.store(true, std::memory_order_relaxed);
            stop_search.store(true, std::memory_order_relaxed);
            return;
        }
    }

    const GlueState& S = path_states[depth];
    if (S.block_count == limit) { record_solution(S, limit); return; }

    Candidate cand = get_best_candidate(S);
    if (cand.p == -1) return;

    bool should_donate = false;
    if (depth < limit - 4 && idle_threads.load(std::memory_order_relaxed) > 0) {
        if (omp_test_lock(&pool_lock)) {
            if (global_task_pool.size() < 10000) { should_donate = true; }
            omp_unset_lock(&pool_lock);
        }
    }

    int p = cand.p; int m = cand.m; int new_id = S.block_count;
    int branches[14] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    if (!FIND_ALL_MODE) {
        for (int i = 13; i > 0; --i) {
            int j = fast_rand(rng_state) % (i + 1); std::swap(branches[i], branches[j]);
        }
    }

    bool has_kept_one = false;

    for (int i = 0; i < 14; ++i) {
        if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

        int n = branches[i];
        GlueState next_S = S;
        next_S.block_count++;
        next_S.block_type_cw[new_id] = (n >= 7);
        int connect_side = (n < 7) ? n : n - 7;

        zip_queue.clear();
        zip_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

        if (perform_zip_and_check(next_S, zip_queue)) {
            if (should_donate) {
                if (!has_kept_one) {
                    path_states[depth + 1] = next_S;
                    has_kept_one = true;
                }
                else {
                    omp_set_lock(&pool_lock);
                    global_task_pool.push_back(next_S);
                    omp_unset_lock(&pool_lock);
                }
            }
            else {
                path_states[depth + 1] = next_S;
                worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
            }
        }
    }

    if (should_donate && has_kept_one) {
        worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
    }
}

void build_encoded_lookup(const mxArray* mxData, int rows, int cols, std::vector<uint64_t>& table, bool is_edge) {
    double* pr = mxGetPr(mxData);
    table.clear(); table.reserve(rows);
    std::vector<int8_t> temp_row(cols);
    for (int r = 0; r < rows; ++r) {
        bool all_zero = true;
        for (int c = 0; c < cols; ++c) {
            temp_row[c] = (int8_t)std::round(pr[r + c * rows]);
            if (temp_row[c] != 0) all_zero = false;
        }
        if (all_zero) continue;
        if (is_edge) {
            int pos_idx = -1, neg_idx = -1;
            for (int c = 0; c < cols; ++c) {
                if (temp_row[c] == 1) pos_idx = c;
                if (temp_row[c] == -1) neg_idx = c;
            }
            if (pos_idx != -1 && neg_idx != -1) {
                std::fill(temp_row.begin(), temp_row.end(), 0);
                temp_row[std::min(pos_idx, neg_idx)] = 1; temp_row[std::max(pos_idx, neg_idx)] = -1;
            }
            table.push_back(encode_edge_vec(temp_row.data(), cols));
        }
        else {
            table.push_back(encode_angle_vec(temp_row.data(), cols));
        }
    }
    std::sort(table.begin(), table.end());
    table.erase(std::unique(table.begin(), table.end()), table.end());
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 4) mexErrMsgTxt("Usage: heptagon_search(...)");

    int Target_N = (int)mxGetScalar(prhs[0]);
    if (Target_N > MAX_N) mexErrMsgIdAndTxt("Heptagon:TargetTooLarge", "Target_N exceeds MAX_N");

    const mxArray* mxE = prhs[1];
    build_encoded_lookup(mxE, (int)mxGetM(mxE), (int)mxGetN(mxE), LOOKUP_E_ENC, true);
    const mxArray* mxA = prhs[2];
    GLOBAL_COLS_A = (int)mxGetN(mxA);
    build_encoded_lookup(mxA, (int)mxGetM(mxA), GLOBAL_COLS_A, LOOKUP_A_ENC, false);

    // 在单线程区一次性初始化静态 O(1) 查找表
    for (int u = 0; u < 7; ++u) {
        for (int v = 0; v < 7; ++v) {
            if (u == v) {
                ALLOWED_E[u][v] = true;
            }
            else {
                int min_s = std::min(u, v);
                int max_s = std::max(u, v);
                uint64_t code = 0;
                for (int i = 0; i < 7; ++i) {
                    int val = 1;
                    if (i == min_s) val = 2; else if (i == max_s) val = 0;
                    code = code * 3 + val;
                }
                auto it = std::lower_bound(LOOKUP_E_ENC.begin(), LOOKUP_E_ENC.end(), code);
                ALLOWED_E[u][v] = (it != LOOKUP_E_ENC.end() && *it == code);
            }
        }
    }

    for (int a1 = 0; a1 < 8; ++a1) {
        for (int a2 = 0; a2 < 8; ++a2) {
            for (int a3 = 0; a3 < 8; ++a3) {
                uint64_t code = 0; int8_t temp[32] = { 0 };
                if (a1 < GLOBAL_COLS_A - 1 && a1 < 31) temp[a1]++;
                if (a2 < GLOBAL_COLS_A - 1 && a2 < 31) temp[a2]++;
                if (a3 < GLOBAL_COLS_A - 1 && a3 < 31) temp[a3]++;
                if (GLOBAL_COLS_A >= 8 && 7 < 32) temp[7] = 2;
                for (int i = 0; i < GLOBAL_COLS_A && i < 32; ++i) { code = (code << 3) | (temp[i] & 0x7); }
                auto it = std::lower_bound(LOOKUP_A_ENC.begin(), LOOKUP_A_ENC.end(), code);
                ALLOWED_A[a1][a2][a3] = (it != LOOKUP_A_ENC.end() && *it == code);
            }
        }
    }

    FIND_ALL_MODE = (mxGetScalar(prhs[3]) > 0.5);
    TARGET_COUNT_LIMIT = (nrhs >= 5) ? (int)mxGetScalar(prhs[4]) : 1;

    // 读取可选的第 6 个超时控制参数（默认值 600 秒）
    if (nrhs >= 6) {
        MAX_DURATION_SECONDS = mxGetScalar(prhs[5]);
    }
    else {
        MAX_DURATION_SECONDS = 600.0;
    }

    global_solutions.clear();
    stop_search.store(false);
    found_count.store(0);
    global_task_pool.clear();
    idle_threads.store(0);
    omp_init_lock(&pool_lock);

    // 初始化运行计时起点
    is_timeout.store(false);
    global_start_time = std::chrono::steady_clock::now();

    // =========================================================
    // Phase 1: 基础广度展开，生成第一批种子启动池
    // =========================================================
    mexPrintf(">> [Phase 1] Adaptive Breadth Splitting...\n");

    GlueState initial_state;
    initial_state.block_count = 1;
    global_task_pool.push_back(initial_state);

    int current_depth = 1;
    while (global_task_pool.size() < MIN_FRAGMENTS && current_depth < Target_N) {
        if (stop_search.load(std::memory_order_relaxed)) break;

        // 阶段 1：自适应分裂过程中的超时拦截
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - global_start_time).count() > MAX_DURATION_SECONDS) {
            is_timeout.store(true, std::memory_order_relaxed);
            stop_search.store(true, std::memory_order_relaxed);
            break;
        }

        std::vector<GlueState> next_pool;

#pragma omp parallel
        {
            std::vector<GlueState> local_next;
            local_next.reserve(1000);
            std::vector<std::pair<EdgeInfo, EdgeInfo>> local_queue;
            local_queue.reserve(64);

#pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < (int)global_task_pool.size(); ++i) {
                if (stop_search.load(std::memory_order_relaxed)) continue;
                const GlueState& S = global_task_pool[i];
                Candidate cand = get_best_candidate(S);
                if (cand.p == -1) continue;

                int p = cand.p; int m = cand.m; int new_id = S.block_count;
                for (int n = 0; n < 14; ++n) {
                    GlueState next_S = S;
                    next_S.block_count++;
                    next_S.block_type_cw[new_id] = (n >= 7);
                    int connect_side = (n < 7) ? n : n - 7;
                    local_queue.clear();
                    local_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

                    if (perform_zip_and_check(next_S, local_queue)) {
                        if (next_S.block_count == Target_N) record_solution(next_S, Target_N);
                        else local_next.push_back(next_S);
                    }
                }
            }
#pragma omp critical(merge_pool)
            { next_pool.insert(next_pool.end(), local_next.begin(), local_next.end()); }
        }
        global_task_pool = std::move(next_pool);
        current_depth++;
        if (global_task_pool.size() > MAX_FRAGMENTS) break;
    }

    if (!FIND_ALL_MODE && !global_task_pool.empty()) {
        srand((unsigned int)time(NULL));
        std::random_shuffle(global_task_pool.begin(), global_task_pool.end());
    }

    if (!is_timeout.load()) {
        mexPrintf(">> Generated %d seeds. [Phase 2] Dynamic Work-Stealing Activated!\n", global_task_pool.size());
    }

    // =========================================================
    // Phase 2: 动态多核窃取引擎
    // =========================================================
#pragma omp parallel
    {
#pragma omp single
        { num_total_threads = omp_get_num_threads(); }

        std::vector<GlueState> thread_path(Target_N + 5);
        std::vector<std::pair<EdgeInfo, EdgeInfo>> thread_zip_queue;
        thread_zip_queue.reserve(64);
        uint32_t thread_rng = 123456789 ^ omp_get_thread_num() ^ (unsigned int)time(NULL);

        while (!stop_search.load(std::memory_order_relaxed)) {
            GlueState task;
            bool got_task = false;

            omp_set_lock(&pool_lock);
            if (!global_task_pool.empty()) {
                task = global_task_pool.back();
                global_task_pool.pop_back();
                got_task = true;
            }
            omp_unset_lock(&pool_lock);

            if (got_task) {
                thread_path[task.block_count] = task;
                worker_dfs_stealing(task.block_count, thread_path, thread_zip_queue, Target_N, thread_rng);
            }
            else {
                idle_threads.fetch_add(1, std::memory_order_relaxed);

                while (global_task_pool.empty() && !stop_search.load(std::memory_order_relaxed)) {
                    if (idle_threads.load(std::memory_order_relaxed) == num_total_threads) {
                        stop_search.store(true, std::memory_order_relaxed);
                        break;
                    }
                    std::this_thread::yield();
                }

                idle_threads.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    omp_destroy_lock(&pool_lock);

    // =========================================================
    // 返回数据 (包含 timeout 布尔标志位)
    // =========================================================
    int count = (int)global_solutions.size();
    const char* fields[] = { "count", "solutions", "timeout" };
    plhs[0] = mxCreateStructMatrix(1, 1, 3, fields);

    mxSetField(plhs[0], 0, "count", mxCreateDoubleScalar((double)count));
    mxSetField(plhs[0], 0, "timeout", mxCreateLogicalScalar(is_timeout.load()));

    mxArray* out_cell = mxCreateCellMatrix(1, count);
    for (int i = 0; i < count; ++i) {
        mxArray* m = mxCreateNumericMatrix(Target_N, 7, mxINT32_CLASS, mxREAL);
        int* d = (int*)mxGetData(m);
        for (int k = 0; k < Target_N * 7; ++k) d[k] = global_solutions[i].flat_data[k];
        mxSetCell(out_cell, i, m);
    }
    mxSetField(plhs[0], 0, "solutions", out_cell);
}*/


// 优化版多核调度 - 支持手动短路跳过边/角验证 + 2阶顶点子集预判剪枝优化 (384核大规模并发优化稳定版)
/*#include "mex.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <thread>
#include <random>
#include <ctime> 
#include <chrono>

// ==========================================================
// 基础配置与全局开关
// ==========================================================
const int MAX_N = 100;

bool SKIP_EDGE_CHECK = false;   // 手动控制开关：跳过边验证
bool SKIP_ANGLE_CHECK = false;  // 手动控制开关：跳过角验证

inline int get_real_side(int s) { return s % 7; }
inline int get_prev_side(int s) { return (s < 7) ? (s + 6) % 7 : 7 + (s - 7 + 6) % 7; }
inline int get_next_side(int s) { return (s < 7) ? (s + 1) % 7 : 7 + (s - 7 + 1) % 7; }
inline int get_angle_val(int s1, int s2) {
    int r1 = s1 % 7; int r2 = s2 % 7;
    if ((r1 == 0 && r2 == 6) || (r1 == 6 && r2 == 0)) return 0;
    return std::max(r1, r2);
}

std::vector<uint64_t> LOOKUP_E_ENC;
std::vector<uint64_t> LOOKUP_A_ENC;
int GLOBAL_COLS_A = 8;

// ==========================================================
// 超快 O(1) 全局查找表
// ==========================================================
bool VALID_2_DEGREE[8][8] = { false };
bool VALID_3_DEGREE[8][8][8] = { false };
bool VALID_EDGE_TRANSITION[7][7] = { false };

struct EdgeInfo {
    int16_t block_id;
    int8_t side;
};

struct GlueState {
    EdgeInfo adj[MAX_N][7];
    int16_t block_count;
    bool block_type_cw[MAX_N];

    GlueState() {
        block_count = 0;
        std::memset(block_type_cw, 0, MAX_N * sizeof(bool));
        for (int i = 0; i < MAX_N; ++i) {
            for (int j = 0; j < 7; ++j) {
                adj[i][j].block_id = -1;
                adj[i][j].side = -1;
            }
        }
    }
};

inline void copy_state(GlueState& dest, const GlueState& src) {
    dest.block_count = src.block_count;
    std::memcpy(dest.adj, src.adj, src.block_count * sizeof(src.adj[0]));
    std::memcpy(dest.block_type_cw, src.block_type_cw, src.block_count * sizeof(bool));

    for (int j = 0; j < 7; ++j) {
        dest.adj[src.block_count][j].block_id = -1;
        dest.adj[src.block_count][j].side = -1;
    }
}

inline uint64_t encode_edge_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = code * 3 + (row[i] + 1);
    return code;
}
inline uint64_t encode_angle_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = (code << 3) | (row[i] & 0x7);
    return code;
}

// 跨平台 CPU 挂起指令（防止高并发自旋锁过热）
inline void cpu_pause() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
    _mm_pause();
#else
    __builtin_ia32_pause();
#endif
#elif defined(__arm__) || defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

void init_all_lookups(const mxArray* mxA, int cols_A) {
    int rows_A = (int)mxGetM(mxA);
    double* pr_A = mxGetPr(mxA);

    std::memset(VALID_2_DEGREE, 0, sizeof(VALID_2_DEGREE));
    for (int r = 0; r < rows_A; ++r) {
        int counts[7] = { 0 };
        for (int c = 0; c < 7 && c < cols_A - 1; ++c) {
            counts[c] = (int)std::round(pr_A[r + c * rows_A]);
        }
        std::vector<int> angles;
        for (int i = 0; i < 7; ++i) {
            for (int k = 0; k < counts[i]; ++k) {
                angles.push_back(i);
            }
        }
        if (angles.size() == 3) {
            int a = angles[0]; int b = angles[1]; int c = angles[2];
            if (a < 8 && b < 8 && c < 8) {
                VALID_2_DEGREE[a][b] = true; VALID_2_DEGREE[b][a] = true;
                VALID_2_DEGREE[b][c] = true; VALID_2_DEGREE[c][b] = true;
                VALID_2_DEGREE[a][c] = true; VALID_2_DEGREE[c][a] = true;
            }
        }
    }

    std::memset(VALID_3_DEGREE, 0, sizeof(VALID_3_DEGREE));
    for (int a1 = 0; a1 < 8; ++a1) {
        for (int a2 = 0; a2 < 8; ++a2) {
            for (int a3 = 0; a3 < 8; ++a3) {
                uint64_t code = 0; int8_t temp[32] = { 0 };
                if (a1 < cols_A - 1) temp[a1]++;
                if (a2 < cols_A - 1) temp[a2]++;
                if (a3 < cols_A - 1) temp[a3]++;
                if (cols_A >= 8) temp[7] = 2;
                for (int i = 0; i < cols_A; ++i) { code = (code << 3) | (temp[i] & 0x7); }
                auto it = std::lower_bound(LOOKUP_A_ENC.begin(), LOOKUP_A_ENC.end(), code);
                if (it != LOOKUP_A_ENC.end() && *it == code) {
                    VALID_3_DEGREE[a1][a2][a3] = true;
                }
            }
        }
    }

    std::memset(VALID_EDGE_TRANSITION, 0, sizeof(VALID_EDGE_TRANSITION));
    for (int u = 0; u < 7; ++u) {
        for (int v = 0; v < 7; ++v) {
            if (u == v) {
                VALID_EDGE_TRANSITION[u][v] = true;
            }
            else {
                int min_s = std::min(u, v); int max_s = std::max(u, v);
                uint64_t code = 0;
                for (int i = 0; i < 7; ++i) {
                    int val = 1;
                    if (i == min_s) val = 2; else if (i == max_s) val = 0;
                    code = code * 3 + val;
                }
                auto it = std::lower_bound(LOOKUP_E_ENC.begin(), LOOKUP_E_ENC.end(), code);
                VALID_EDGE_TRANSITION[u][v] = (it != LOOKUP_E_ENC.end() && *it == code);
            }
        }
    }
}

struct SolutionData {
    int flat_data[MAX_N * 7];
    SolutionData() { std::memset(flat_data, 0, sizeof(flat_data)); }
};

// ==========================================================
// 全局状态与动态负载均衡控制（高并发深度优化）
// ==========================================================
std::vector<SolutionData> global_solutions;
std::atomic<bool> stop_search(false);
std::atomic<int> found_count(0);
bool FIND_ALL_MODE = false;
int TARGET_COUNT_LIMIT = 1;

std::vector<GlueState> global_task_pool;
std::atomic<int> global_pool_size(0); // 【关键优化】无锁原子计数器，规避 384 核锁风暴
omp_lock_t pool_lock;
std::atomic<int> idle_threads(0);
int num_total_threads = 1;

inline uint32_t fast_rand(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

inline bool check_edge_lookup(int side_p, int side_q) {
    if (SKIP_EDGE_CHECK) return true;
    return VALID_EDGE_TRANSITION[get_real_side(side_p)][get_real_side(side_q)];
}

inline bool check_angle_lookup(int a1, int a2, int a3) {
    if (SKIP_ANGLE_CHECK) return true;
    if (a1 >= 8 || a2 >= 8 || a3 >= 8 || a1 < 0 || a2 < 0 || a3 < 0) return false;
    return VALID_3_DEGREE[a1][a2][a3];
}

bool perform_zip_and_check(GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& queue) {
    size_t head = 0;
    while (head < queue.size()) {
        auto pair = queue[head++];
        EdgeInfo e1 = pair.first; EdgeInfo e2 = pair.second;
        if (!check_edge_lookup(e1.side, e2.side)) return false;

        int p = e1.block_id; int ps = get_real_side(e1.side);
        int q = e2.block_id; int qs = get_real_side(e2.side);

        if (S.adj[p][ps].block_id != -1) {
            if (S.adj[p][ps].block_id != q) return false;
            continue;
        }
        if (S.adj[q][qs].block_id != -1) {
            if (S.adj[q][qs].block_id != p) return false;
            continue;
        }

        S.adj[p][ps] = { (int16_t)q, (int8_t)e2.side }; S.adj[q][qs] = { (int16_t)p, (int8_t)e1.side };
        bool same_dir = (S.block_type_cw[p] == S.block_type_cw[q]);
        int p_neighbors[2] = { get_prev_side(e1.side), get_next_side(e1.side) };
        int q_neighbors[2];
        if (same_dir) {
            q_neighbors[0] = get_next_side(e2.side); q_neighbors[1] = get_prev_side(e2.side);
        }
        else {
            q_neighbors[0] = get_prev_side(e2.side); q_neighbors[1] = get_next_side(e2.side);
        }

        for (int i = 0; i < 2; ++i) {
            int s_p_check = p_neighbors[i]; int s_q_check = q_neighbors[i];

            if (!SKIP_ANGLE_CHECK) {
                int a1 = get_angle_val(s_p_check, e1.side);
                int a2 = get_angle_val(e2.side, s_q_check);
                if (a1 < 8 && a2 < 8 && !VALID_2_DEGREE[a1][a2]) {
                    return false;
                }
            }

            EdgeInfo link_p = S.adj[p][get_real_side(s_p_check)];
            EdgeInfo link_q = S.adj[q][get_real_side(s_q_check)];
            int k = -1;

            if (link_p.block_id != -1 && link_q.block_id != -1) {
                if (link_p.block_id != link_q.block_id) return false;
                k = link_p.block_id;
                int a1 = get_angle_val(s_p_check, e1.side); int a2 = get_angle_val(e2.side, s_q_check); int a3 = get_angle_val(link_p.side, link_q.side);
                if (!check_angle_lookup(a1, a2, a3)) return false;
            }
            else if (link_p.block_id != -1 && link_q.block_id == -1) {
                k = link_p.block_id; if (k == q) continue;
                bool same_dir_pk = (S.block_type_cw[p] == S.block_type_cw[k]); bool is_p_next = (s_p_check == get_next_side(e1.side));
                int s_k_target = same_dir_pk ? (is_p_next ? get_next_side(link_p.side) : get_prev_side(link_p.side)) : (is_p_next ? get_prev_side(link_p.side) : get_next_side(link_p.side));
                queue.push_back({ EdgeInfo{ (int16_t)q, (int8_t)s_q_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
            else if (link_q.block_id != -1 && link_p.block_id == -1) {
                k = link_q.block_id; if (k == p) continue;
                bool same_dir_qk = (S.block_type_cw[q] == S.block_type_cw[k]); bool is_q_next = (s_q_check == get_next_side(e2.side));
                int s_k_target = same_dir_qk ? (is_q_next ? get_next_side(link_q.side) : get_prev_side(link_q.side)) : (is_q_next ? get_prev_side(link_q.side) : get_next_side(link_q.side));
                queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)s_p_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
        }
    }
    return true;
}

struct Candidate {
    int p;
    int m;
    int score;
    bool is_dead_end;
    int num_valid_branches;
    int valid_branches[14];
};

Candidate get_best_candidate(const GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& temp_queue) {
    Candidate result = { -1, -1, -1, false, 0, {0} };

    int min_i = -1;
    for (int i = 0; i < S.block_count; ++i) {
        bool has_open = false;
        for (int j = 0; j < 7; ++j) {
            if (S.adj[i][j].block_id == -1) {
                has_open = true;
                break;
            }
        }
        if (has_open) {
            min_i = i;
            break;
        }
    }

    if (min_i == -1) return result;

    int max_score = -1;
    std::vector<int> candidates_j;
    candidates_j.reserve(7);

    for (int j = 0; j < 7; ++j) {
        if (S.adj[min_i][j].block_id == -1) {
            int score = 0;
            if (S.adj[min_i][get_real_side(get_prev_side(j))].block_id != -1) score += 10;
            if (S.adj[min_i][get_real_side(get_next_side(j))].block_id != -1) score += 10;

            if (score > max_score) {
                max_score = score;
                candidates_j.clear();
                candidates_j.push_back(j);
            }
            else if (score == max_score) {
                candidates_j.push_back(j);
            }
        }
    }

    int best_j = -1;
    int min_valid_branches = 999;
    int best_branches[14];
    int best_branches_count = 0;

    GlueState next_S;

    for (int j : candidates_j) {
        int valid_branches[14];
        int valid_count = 0;

        for (int n = 0; n < 14; ++n) {
            copy_state(next_S, S);
            next_S.block_count++;
            int new_id = S.block_count;
            next_S.block_type_cw[new_id] = (n >= 7);
            int connect_side = (n < 7) ? n : n - 7;

            temp_queue.clear();
            temp_queue.push_back({ EdgeInfo{ (int16_t)min_i, (int8_t)j }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

            if (perform_zip_and_check(next_S, temp_queue)) {
                valid_branches[valid_count++] = n;
            }
        }

        if (valid_count == 0) {
            result.is_dead_end = true;
            return result;
        }

        if (valid_count < min_valid_branches) {
            min_valid_branches = valid_count;
            best_j = j;
            best_branches_count = valid_count;
            std::memcpy(best_branches, valid_branches, valid_count * sizeof(int));
        }
    }

    result.p = min_i;
    result.m = best_j;
    result.score = max_score;
    result.num_valid_branches = best_branches_count;
    std::memcpy(result.valid_branches, best_branches, best_branches_count * sizeof(int));

    return result;
}

void record_solution(const GlueState& S, int limit) {
    SolutionData sol;
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < 7; ++j) {
            sol.flat_data[i + j * limit] = (S.adj[i][j].block_id != -1) ? (S.adj[i][j].block_id + 1) * 100 + S.adj[i][j].side : 0;
        }
    }
#pragma omp critical(save_sol)
    {
        if (FIND_ALL_MODE || found_count.load(std::memory_order_relaxed) < TARGET_COUNT_LIMIT) {
            global_solutions.push_back(sol);
            if (!FIND_ALL_MODE && ++found_count >= TARGET_COUNT_LIMIT) {
                stop_search.store(true, std::memory_order_relaxed);
            }
        }
    }
}

// Phase 2: 动态任务窃取深度优先引擎（384核低锁争用深度优化版）
void worker_dfs_stealing(int depth, std::vector<GlueState>& path_states, std::vector<std::pair<EdgeInfo, EdgeInfo>>& zip_queue, int limit, uint32_t& rng_state) {
    if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

    const GlueState& S = path_states[depth];
    if (S.block_count == limit) { record_solution(S, limit); return; }

    Candidate cand = get_best_candidate(S, zip_queue);
    if (cand.is_dead_end || cand.p == -1) return;

    bool should_donate = false;
    if (depth < limit - 4 && idle_threads.load(std::memory_order_relaxed) > 0) {
        if (omp_test_lock(&pool_lock)) {
            if (global_task_pool.size() < 20000) { should_donate = true; }
            omp_unset_lock(&pool_lock);
        }
    }

    int p = cand.p; int m = cand.m; int new_id = S.block_count;
    int num_branches = cand.num_valid_branches;
    int branches[14];
    std::memcpy(branches, cand.valid_branches, num_branches * sizeof(int));

    if (!FIND_ALL_MODE) {
        for (int i = num_branches - 1; i > 0; --i) {
            int j = fast_rand(rng_state) % (i + 1); std::swap(branches[i], branches[j]);
        }
    }

    std::vector<GlueState> valid_children;
    if (should_donate) valid_children.reserve(num_branches);

    for (int i = 0; i < num_branches; ++i) {
        if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

        int n = branches[i];

        copy_state(path_states[depth + 1], S);
        GlueState& next_S = path_states[depth + 1];
        next_S.block_count++;
        next_S.block_type_cw[new_id] = (n >= 7);
        int connect_side = (n < 7) ? n : n - 7;

        zip_queue.clear();
        zip_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

        if (perform_zip_and_check(next_S, zip_queue)) {
            if (should_donate) {
                valid_children.push_back(next_S);
            }
            else {
                worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
            }
        }
    }

    if (should_donate && !valid_children.empty()) {
        omp_set_lock(&pool_lock);
        int added = 0;
        for (size_t i = 1; i < valid_children.size(); ++i) {
            global_task_pool.push_back(valid_children[i]);
            added++;
        }
        omp_unset_lock(&pool_lock);
        if (added > 0) {
            global_pool_size.fetch_add(added, std::memory_order_release);
        }

        copy_state(path_states[depth + 1], valid_children[0]);
        worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
    }
}

void build_encoded_lookup(const mxArray* mxData, int rows, int cols, std::vector<uint64_t>& table, bool is_edge) {
    double* pr = mxGetPr(mxData);
    table.clear(); table.reserve(rows);
    std::vector<int8_t> temp_row(cols);
    for (int r = 0; r < rows; ++r) {
        bool all_zero = true;
        for (int c = 0; c < cols; ++c) {
            temp_row[c] = (int8_t)std::round(pr[r + c * rows]);
            if (temp_row[c] != 0) all_zero = false;
        }
        if (all_zero) continue;
        if (is_edge) {
            int pos_idx = -1, neg_idx = -1;
            for (int c = 0; c < cols; ++c) {
                if (temp_row[c] == 1) pos_idx = c;
                if (temp_row[c] == -1) neg_idx = c;
            }
            if (pos_idx != -1 && neg_idx != -1) {
                std::fill(temp_row.begin(), temp_row.end(), 0);
                temp_row[std::min(pos_idx, neg_idx)] = 1; temp_row[std::max(pos_idx, neg_idx)] = -1;
            }
            table.push_back(encode_edge_vec(temp_row.data(), cols));
        }
        else {
            table.push_back(encode_angle_vec(temp_row.data(), cols));
        }
    }
    std::sort(table.begin(), table.end());
    table.erase(std::unique(table.begin(), table.end()), table.end());
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 4) mexErrMsgTxt("Usage: heptagon_search(...)");

    int Target_N = (int)mxGetScalar(prhs[0]);
    if (Target_N > MAX_N) mexErrMsgIdAndTxt("Heptagon:TargetTooLarge", "Target_N exceeds MAX_N");

    const mxArray* mxE = prhs[1];
    build_encoded_lookup(mxE, (int)mxGetM(mxE), (int)mxGetN(mxE), LOOKUP_E_ENC, true);
    const mxArray* mxA = prhs[2];
    GLOBAL_COLS_A = (int)mxGetN(mxA);
    build_encoded_lookup(mxA, (int)mxGetM(mxA), GLOBAL_COLS_A, LOOKUP_A_ENC, false);

    init_all_lookups(mxA, GLOBAL_COLS_A);

    FIND_ALL_MODE = (mxGetScalar(prhs[3]) > 0.5);
    TARGET_COUNT_LIMIT = (nrhs >= 5) ? (int)mxGetScalar(prhs[4]) : 1;

    SKIP_EDGE_CHECK = (nrhs >= 6) ? (mxGetScalar(prhs[5]) > 0.5) : false;
    SKIP_ANGLE_CHECK = (nrhs >= 7) ? (mxGetScalar(prhs[6]) > 0.5) : false;

    if (SKIP_EDGE_CHECK)  mexPrintf(">> [System Notice] Edge checks are BYPASSED.\n");
    if (SKIP_ANGLE_CHECK) mexPrintf(">> [System Notice] Angle checks are BYPASSED.\n");

    global_solutions.clear();
    stop_search.store(false);
    found_count.store(0);
    global_task_pool.clear();
    global_pool_size.store(0);
    idle_threads.store(0);
    omp_init_lock(&pool_lock);

    // =========================================================
    // 【384核自适应分裂控制】
    // =========================================================
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads < 1) num_threads = 1;

    size_t min_fragments;
    size_t max_fragments;

    if (FIND_ALL_MODE) {
        // 每个核心保持分配 50+ 个种子
        min_fragments = std::max((size_t)2000, (size_t)num_threads * 50);
        max_fragments = min_fragments * 6;
    }
    else {
        // 核心多时，适当限制种子过度展开，保证单解快速收敛
        min_fragments = std::max((size_t)120, (size_t)num_threads * 8);
        max_fragments = min_fragments * 4;
    }

    mexPrintf(">> [Phase 1] Hardware-Adaptive Splitting (Detected Cores: %d, Target Seeds: %d)...\n",
        num_threads, (int)min_fragments);

    GlueState initial_state;
    initial_state.block_count = 1;
    global_task_pool.push_back(initial_state);
    global_pool_size.store(1, std::memory_order_relaxed);

    int current_depth = 1;
    while (global_task_pool.size() < min_fragments && current_depth < Target_N) {
        if (stop_search.load(std::memory_order_relaxed)) break;
        std::vector<GlueState> next_pool;

#pragma omp parallel
        {
            std::vector<GlueState> local_next;
            local_next.reserve(1000);
            std::vector<std::pair<EdgeInfo, EdgeInfo>> local_queue;
            local_queue.reserve(64);

            // 【关键优化】：将 chunk_size 降为 1（schedule(dynamic, 1)），保证高核心利用率
#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < (int)global_task_pool.size(); ++i) {
                if (stop_search.load(std::memory_order_relaxed)) continue;
                const GlueState& S = global_task_pool[i];

                Candidate cand = get_best_candidate(S, local_queue);
                if (cand.is_dead_end || cand.p == -1) continue;

                int p = cand.p; int m = cand.m; int new_id = S.block_count;
                for (int b = 0; b < cand.num_valid_branches; ++b) {
                    int n = cand.valid_branches[b];

                    GlueState next_S;
                    copy_state(next_S, S);
                    next_S.block_count++;
                    next_S.block_type_cw[new_id] = (n >= 7);
                    int connect_side = (n < 7) ? n : n - 7;
                    local_queue.clear();
                    local_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

                    if (perform_zip_and_check(next_S, local_queue)) {
                        if (next_S.block_count == Target_N) record_solution(next_S, Target_N);
                        else local_next.push_back(next_S);
                    }
                }
            }
#pragma omp critical(merge_pool)
            { next_pool.insert(next_pool.end(), local_next.begin(), local_next.end()); }
        }
        global_task_pool = std::move(next_pool);
        global_pool_size.store((int)global_task_pool.size(), std::memory_order_relaxed);
        current_depth++;
        if (global_task_pool.size() > max_fragments) break;
    }

    if (!FIND_ALL_MODE && !global_task_pool.empty()) {
        std::mt19937 g(static_cast<unsigned int>(time(NULL)));
        std::shuffle(global_task_pool.begin(), global_task_pool.end(), g);
    }

    mexPrintf(">> Generated %d seeds. [Phase 2] Dynamic Work-Stealing Activated!\n", (int)global_task_pool.size());

    // =========================================================
    // Phase 2: 动态多核窃取引擎（384核大规模并发优化版）
    // =========================================================
#pragma omp parallel
    {
#pragma omp single
        { num_total_threads = omp_get_num_threads(); }

        std::vector<GlueState> thread_path(Target_N + 5);
        std::vector<std::pair<EdgeInfo, EdgeInfo>> thread_zip_queue;
        thread_zip_queue.reserve(64);
        uint32_t thread_rng = 123456789 ^ omp_get_thread_num() ^ (unsigned int)time(NULL);

        while (!stop_search.load(std::memory_order_relaxed)) {
            GlueState task;
            bool got_task = false;

            // 【关键无锁双重检查】：规避384核高频空拿锁造成的总线堵塞
            if (global_pool_size.load(std::memory_order_relaxed) > 0) {
                omp_set_lock(&pool_lock);
                if (!global_task_pool.empty()) {
                    task = global_task_pool.back();
                    global_task_pool.pop_back();
                    global_pool_size.fetch_sub(1, std::memory_order_release);
                    got_task = true;
                }
                omp_unset_lock(&pool_lock);
            }

            if (got_task) {
                copy_state(thread_path[task.block_count], task);
                worker_dfs_stealing(task.block_count, thread_path, thread_zip_queue, Target_N, thread_rng);
            }
            else {
                idle_threads.fetch_add(1, std::memory_order_relaxed);

                int spin_count = 0;
                while (global_pool_size.load(std::memory_order_relaxed) == 0 && !stop_search.load(std::memory_order_relaxed)) {
                    if (idle_threads.load(std::memory_order_relaxed) == num_total_threads) {
                        stop_search.store(true, std::memory_order_relaxed);
                        break;
                    }

                    // 【关键优化】：空闲线程自适应温和等待，消除384核调度器上下文切换暴涨
                    if (spin_count++ < 100) {
                        cpu_pause();
                    }
                    else {
                        std::this_thread::sleep_for(std::chrono::microseconds(50)); // 轻微休眠，释放操作系统时间片
                    }
                }

                idle_threads.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    omp_destroy_lock(&pool_lock);

    // =========================================================
    // 返回数据
    // =========================================================
    int count = (int)global_solutions.size();
    const char* fields[] = { "count", "solutions" };
    plhs[0] = mxCreateStructMatrix(1, 1, 2, fields);
    mxSetField(plhs[0], 0, "count", mxCreateDoubleScalar((double)count));

    mxArray* out_cell = mxCreateCellMatrix(1, count);
    for (int i = 0; i < count; ++i) {
        mxArray* m = mxCreateNumericMatrix(Target_N, 7, mxINT32_CLASS, mxREAL);
        int* d = (int*)mxGetData(m);
        for (int k = 0; k < Target_N * 7; ++k) d[k] = global_solutions[i].flat_data[k];
        mxSetCell(out_cell, i, m);
    }
    mxSetField(plhs[0], 0, "solutions", out_cell);
}*/



// 优化版多核调度 - 支持手动短路跳过边/角验证 + 2阶顶点子集预判剪枝优化 (去通信死锁稳定版)+统计所有块的顺逆时针方向
/*#include "mex.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <thread>
#include <random>
#include <ctime> 

// ==========================================================
// 基础配置与全局开关
// ==========================================================
const int MAX_N = 1000;

bool SKIP_EDGE_CHECK = false;   // 手动控制开关：跳过边验证
bool SKIP_ANGLE_CHECK = false;  // 手动控制开关：跳过角验证

inline int get_real_side(int s) { return s % 7; }
inline int get_prev_side(int s) { return (s < 7) ? (s + 6) % 7 : 7 + (s - 7 + 6) % 7; }
inline int get_next_side(int s) { return (s < 7) ? (s + 1) % 7 : 7 + (s - 7 + 1) % 7; }
inline int get_angle_val(int s1, int s2) {
    int r1 = s1 % 7; int r2 = s2 % 7;
    if ((r1 == 0 && r2 == 6) || (r1 == 6 && r2 == 0)) return 0;
    return std::max(r1, r2);
}

std::vector<uint64_t> LOOKUP_E_ENC;
std::vector<uint64_t> LOOKUP_A_ENC;
int GLOBAL_COLS_A = 8;

// ==========================================================
// 【优化】超快 O(1) 全局查找表，免去运行时二分查找与编码开销
// ==========================================================
bool VALID_2_DEGREE[8][8] = { false };
bool VALID_3_DEGREE[8][8][8] = { false };
bool VALID_EDGE_TRANSITION[7][7] = { false };

struct EdgeInfo {
    int16_t block_id;
    int8_t side;
};

struct GlueState {
    EdgeInfo adj[MAX_N][7];
    int16_t block_count;
    bool block_type_cw[MAX_N];

    GlueState() {
        block_count = 0;
        std::memset(block_type_cw, 0, MAX_N * sizeof(bool));
        for (int i = 0; i < MAX_N; ++i) {
            for (int j = 0; j < 7; ++j) {
                adj[i][j].block_id = -1;
                adj[i][j].side = -1;
            }
        }
    }
};

// 增量状态复制
inline void copy_state(GlueState& dest, const GlueState& src) {
    dest.block_count = src.block_count;
    std::memcpy(dest.adj, src.adj, src.block_count * sizeof(src.adj[0]));
    std::memcpy(dest.block_type_cw, src.block_type_cw, src.block_count * sizeof(bool));

    for (int j = 0; j < 7; ++j) {
        dest.adj[src.block_count][j].block_id = -1;
        dest.adj[src.block_count][j].side = -1;
    }
}

inline uint64_t encode_edge_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = code * 3 + (row[i] + 1);
    return code;
}
inline uint64_t encode_angle_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = (code << 3) | (row[i] & 0x7);
    return code;
}

// 初始化全局高速过滤表
void init_all_lookups(const mxArray* mxA, int cols_A) {
    int rows_A = (int)mxGetM(mxA);
    double* pr_A = mxGetPr(mxA);

    std::memset(VALID_2_DEGREE, 0, sizeof(VALID_2_DEGREE));
    for (int r = 0; r < rows_A; ++r) {
        int counts[7] = { 0 };
        for (int c = 0; c < 7 && c < cols_A - 1; ++c) {
            counts[c] = (int)std::round(pr_A[r + c * rows_A]);
        }
        std::vector<int> angles;
        for (int i = 0; i < 7; ++i) {
            for (int k = 0; k < counts[i]; ++k) {
                angles.push_back(i);
            }
        }
        if (angles.size() == 3) {
            int a = angles[0]; int b = angles[1]; int c = angles[2];
            if (a < 8 && b < 8 && c < 8) {
                VALID_2_DEGREE[a][b] = true; VALID_2_DEGREE[b][a] = true;
                VALID_2_DEGREE[b][c] = true; VALID_2_DEGREE[c][b] = true;
                VALID_2_DEGREE[a][c] = true; VALID_2_DEGREE[c][a] = true;
            }
        }
    }

    std::memset(VALID_3_DEGREE, 0, sizeof(VALID_3_DEGREE));
    for (int a1 = 0; a1 < 8; ++a1) {
        for (int a2 = 0; a2 < 8; ++a2) {
            for (int a3 = 0; a3 < 8; ++a3) {
                uint64_t code = 0; int8_t temp[32] = { 0 };
                if (a1 < cols_A - 1) temp[a1]++;
                if (a2 < cols_A - 1) temp[a2]++;
                if (a3 < cols_A - 1) temp[a3]++;
                if (cols_A >= 8) temp[7] = 2;
                for (int i = 0; i < cols_A; ++i) { code = (code << 3) | (temp[i] & 0x7); }
                auto it = std::lower_bound(LOOKUP_A_ENC.begin(), LOOKUP_A_ENC.end(), code);
                if (it != LOOKUP_A_ENC.end() && *it == code) {
                    VALID_3_DEGREE[a1][a2][a3] = true;
                }
            }
        }
    }

    std::memset(VALID_EDGE_TRANSITION, 0, sizeof(VALID_EDGE_TRANSITION));
    for (int u = 0; u < 7; ++u) {
        for (int v = 0; v < 7; ++v) {
            if (u == v) {
                VALID_EDGE_TRANSITION[u][v] = true;
            }
            else {
                int min_s = std::min(u, v); int max_s = std::max(u, v);
                uint64_t code = 0;
                for (int i = 0; i < 7; ++i) {
                    int val = 1;
                    if (i == min_s) val = 2; else if (i == max_s) val = 0;
                    code = code * 3 + val;
                }
                auto it = std::lower_bound(LOOKUP_E_ENC.begin(), LOOKUP_E_ENC.end(), code);
                VALID_EDGE_TRANSITION[u][v] = (it != LOOKUP_E_ENC.end() && *it == code);
            }
        }
    }
}

struct SolutionData {
    int flat_data[MAX_N * 7];
    SolutionData() { std::memset(flat_data, 0, sizeof(flat_data)); }
};

// ==========================================================
// 全局状态与动态负载均衡控制
// ==========================================================
std::vector<SolutionData> global_solutions;
std::atomic<bool> stop_search(false);
std::atomic<int> found_count(0);
bool FIND_ALL_MODE = false;
int TARGET_COUNT_LIMIT = 1;

std::vector<GlueState> global_task_pool;
omp_lock_t pool_lock;
std::atomic<int> idle_threads(0);
int num_total_threads = 1;

inline uint32_t fast_rand(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

inline bool check_edge_lookup(int side_p, int side_q) {
    if (SKIP_EDGE_CHECK) return true;
    return VALID_EDGE_TRANSITION[get_real_side(side_p)][get_real_side(side_q)];
}

inline bool check_angle_lookup(int a1, int a2, int a3) {
    if (SKIP_ANGLE_CHECK) return true;
    if (a1 >= 8 || a2 >= 8 || a3 >= 8 || a1 < 0 || a2 < 0 || a3 < 0) return false;
    return VALID_3_DEGREE[a1][a2][a3];
}

// 拓扑传导与闭合检查
bool perform_zip_and_check(GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& queue) {
    size_t head = 0;
    while (head < queue.size()) {
        auto pair = queue[head++];
        EdgeInfo e1 = pair.first; EdgeInfo e2 = pair.second;
        if (!check_edge_lookup(e1.side, e2.side)) return false;

        int p = e1.block_id; int ps = get_real_side(e1.side);
        int q = e2.block_id; int qs = get_real_side(e2.side);

        if (S.adj[p][ps].block_id != -1) {
            if (S.adj[p][ps].block_id != q) return false;
            continue;
        }
        if (S.adj[q][qs].block_id != -1) {
            if (S.adj[q][qs].block_id != p) return false;
            continue;
        }

        S.adj[p][ps] = { (int16_t)q, (int8_t)e2.side }; S.adj[q][qs] = { (int16_t)p, (int8_t)e1.side };
        bool same_dir = (S.block_type_cw[p] == S.block_type_cw[q]);
        int p_neighbors[2] = { get_prev_side(e1.side), get_next_side(e1.side) };
        int q_neighbors[2];
        if (same_dir) {
            q_neighbors[0] = get_next_side(e2.side); q_neighbors[1] = get_prev_side(e2.side);
        }
        else {
            q_neighbors[0] = get_prev_side(e2.side); q_neighbors[1] = get_next_side(e2.side);
        }

        for (int i = 0; i < 2; ++i) {
            int s_p_check = p_neighbors[i]; int s_q_check = q_neighbors[i];

            if (!SKIP_ANGLE_CHECK) {
                int a1 = get_angle_val(s_p_check, e1.side);
                int a2 = get_angle_val(e2.side, s_q_check);
                if (a1 < 8 && a2 < 8 && !VALID_2_DEGREE[a1][a2]) {
                    return false;
                }
            }

            EdgeInfo link_p = S.adj[p][get_real_side(s_p_check)];
            EdgeInfo link_q = S.adj[q][get_real_side(s_q_check)];
            int k = -1;

            if (link_p.block_id != -1 && link_q.block_id != -1) {
                if (link_p.block_id != link_q.block_id) return false;
                k = link_p.block_id;
                int a1 = get_angle_val(s_p_check, e1.side); int a2 = get_angle_val(e2.side, s_q_check); int a3 = get_angle_val(link_p.side, link_q.side);
                if (!check_angle_lookup(a1, a2, a3)) return false;
            }
            else if (link_p.block_id != -1 && link_q.block_id == -1) {
                k = link_p.block_id; if (k == q) continue;
                bool same_dir_pk = (S.block_type_cw[p] == S.block_type_cw[k]); bool is_p_next = (s_p_check == get_next_side(e1.side));
                int s_k_target = same_dir_pk ? (is_p_next ? get_next_side(link_p.side) : get_prev_side(link_p.side)) : (is_p_next ? get_prev_side(link_p.side) : get_next_side(link_p.side));
                queue.push_back({ EdgeInfo{ (int16_t)q, (int8_t)s_q_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
            else if (link_q.block_id != -1 && link_p.block_id == -1) {
                k = link_q.block_id; if (k == p) continue;
                bool same_dir_qk = (S.block_type_cw[q] == S.block_type_cw[k]); bool is_q_next = (s_q_check == get_next_side(e2.side));
                int s_k_target = same_dir_qk ? (is_q_next ? get_next_side(link_q.side) : get_prev_side(link_q.side)) : (is_q_next ? get_prev_side(link_q.side) : get_next_side(link_q.side));
                queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)s_p_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
        }
    }
    return true;
}

// MRV 候选选择器
struct Candidate {
    int p;
    int m;
    int score;
    bool is_dead_end;
    int num_valid_branches;
    int valid_branches[14];
};

Candidate get_best_candidate(const GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& temp_queue) {
    Candidate result = { -1, -1, -1, false, 0, {0} };

    int min_i = -1;
    for (int i = 0; i < S.block_count; ++i) {
        bool has_open = false;
        for (int j = 0; j < 7; ++j) {
            if (S.adj[i][j].block_id == -1) {
                has_open = true;
                break;
            }
        }
        if (has_open) {
            min_i = i;
            break;
        }
    }

    if (min_i == -1) return result;

    int max_score = -1;
    std::vector<int> candidates_j;
    candidates_j.reserve(7);

    for (int j = 0; j < 7; ++j) {
        if (S.adj[min_i][j].block_id == -1) {
            int score = 0;
            if (S.adj[min_i][get_real_side(get_prev_side(j))].block_id != -1) score += 10;
            if (S.adj[min_i][get_real_side(get_next_side(j))].block_id != -1) score += 10;

            if (score > max_score) {
                max_score = score;
                candidates_j.clear();
                candidates_j.push_back(j);
            }
            else if (score == max_score) {
                candidates_j.push_back(j);
            }
        }
    }

    int best_j = -1;
    int min_valid_branches = 999;
    int best_branches[14];
    int best_branches_count = 0;

    GlueState next_S;

    for (int j : candidates_j) {
        int valid_branches[14];
        int valid_count = 0;

        for (int n = 0; n < 14; ++n) {
            copy_state(next_S, S);
            next_S.block_count++;
            int new_id = S.block_count;
            next_S.block_type_cw[new_id] = (n >= 7);
            int connect_side = (n < 7) ? n : n - 7;

            temp_queue.clear();
            temp_queue.push_back({ EdgeInfo{ (int16_t)min_i, (int8_t)j }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

            if (perform_zip_and_check(next_S, temp_queue)) {
                valid_branches[valid_count++] = n;
            }
        }

        if (valid_count == 0) {
            result.is_dead_end = true;
            return result;
        }

        if (valid_count < min_valid_branches) {
            min_valid_branches = valid_count;
            best_j = j;
            best_branches_count = valid_count;
            std::memcpy(best_branches, valid_branches, valid_count * sizeof(int));
        }
    }

    result.p = min_i;
    result.m = best_j;
    result.score = max_score;
    result.num_valid_branches = best_branches_count;
    std::memcpy(result.valid_branches, best_branches, best_branches_count * sizeof(int));

    return result;
}

void record_solution(const GlueState& S, int limit) {
    SolutionData sol;
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < 7; ++j) {
            if (S.adj[i][j].block_id == -1) {
                sol.flat_data[i + j * limit] = 0;
                continue;
            }

            // Side codes 0..6 mean CCW; 7..13 mean CW.
            // Preserve block_type_cw in the encoded result.
            const int neighbor_id = S.adj[i][j].block_id;
            const int encoded_side = S.adj[i][j].side
                + (S.block_type_cw[neighbor_id] ? 7 : 0);
            sol.flat_data[i + j * limit] = (neighbor_id + 1) * 100 + encoded_side;
        }
    }
#pragma omp critical(save_sol)
    {
        if (FIND_ALL_MODE || found_count.load(std::memory_order_relaxed) < TARGET_COUNT_LIMIT) {
            global_solutions.push_back(sol);
            if (!FIND_ALL_MODE && ++found_count >= TARGET_COUNT_LIMIT) {
                stop_search.store(true, std::memory_order_relaxed);
            }
        }
    }
}

// Phase 2: 动态任务窃取深度优先引擎 (去交互死锁版)
void worker_dfs_stealing(int depth, std::vector<GlueState>& path_states, std::vector<std::pair<EdgeInfo, EdgeInfo>>& zip_queue, int limit, uint32_t& rng_state) {
    if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

    const GlueState& S = path_states[depth];
    if (S.block_count == limit) { record_solution(S, limit); return; }

    Candidate cand = get_best_candidate(S, zip_queue);
    if (cand.is_dead_end || cand.p == -1) return;

    bool should_donate = false;
    if (depth < limit - 4 && idle_threads.load(std::memory_order_relaxed) > 0) {
        if (omp_test_lock(&pool_lock)) {
            if (global_task_pool.size() < 10000) { should_donate = true; }
            omp_unset_lock(&pool_lock);
        }
    }

    int p = cand.p; int m = cand.m; int new_id = S.block_count;
    int num_branches = cand.num_valid_branches;
    int branches[14];
    std::memcpy(branches, cand.valid_branches, num_branches * sizeof(int));

    if (!FIND_ALL_MODE) {
        for (int i = num_branches - 1; i > 0; --i) {
            int j = fast_rand(rng_state) % (i + 1); std::swap(branches[i], branches[j]);
        }
    }

    std::vector<GlueState> valid_children;
    if (should_donate) valid_children.reserve(num_branches);

    for (int i = 0; i < num_branches; ++i) {
        if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

        int n = branches[i];

        copy_state(path_states[depth + 1], S);
        GlueState& next_S = path_states[depth + 1];
        next_S.block_count++;
        next_S.block_type_cw[new_id] = (n >= 7);
        int connect_side = (n < 7) ? n : n - 7;

        zip_queue.clear();
        zip_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

        if (perform_zip_and_check(next_S, zip_queue)) {
            if (should_donate) {
                valid_children.push_back(next_S);
            }
            else {
                worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
            }
        }
    }

    if (should_donate && !valid_children.empty()) {
        omp_set_lock(&pool_lock);
        for (size_t i = 1; i < valid_children.size(); ++i) {
            global_task_pool.push_back(valid_children[i]);
        }
        omp_unset_lock(&pool_lock);

        copy_state(path_states[depth + 1], valid_children[0]);
        worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
    }
}

void build_encoded_lookup(const mxArray* mxData, int rows, int cols, std::vector<uint64_t>& table, bool is_edge) {
    double* pr = mxGetPr(mxData);
    table.clear(); table.reserve(rows);
    std::vector<int8_t> temp_row(cols);
    for (int r = 0; r < rows; ++r) {
        bool all_zero = true;
        for (int c = 0; c < cols; ++c) {
            temp_row[c] = (int8_t)std::round(pr[r + c * rows]);
            if (temp_row[c] != 0) all_zero = false;
        }
        if (all_zero) continue;
        if (is_edge) {
            int pos_idx = -1, neg_idx = -1;
            for (int c = 0; c < cols; ++c) {
                if (temp_row[c] == 1) pos_idx = c;
                if (temp_row[c] == -1) neg_idx = c;
            }
            if (pos_idx != -1 && neg_idx != -1) {
                std::fill(temp_row.begin(), temp_row.end(), 0);
                temp_row[std::min(pos_idx, neg_idx)] = 1; temp_row[std::max(pos_idx, neg_idx)] = -1;
            }
            table.push_back(encode_edge_vec(temp_row.data(), cols));
        }
        else {
            table.push_back(encode_angle_vec(temp_row.data(), cols));
        }
    }
    std::sort(table.begin(), table.end());
    table.erase(std::unique(table.begin(), table.end()), table.end());
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 4) mexErrMsgTxt("Usage: heptagon_search(...)");

    int Target_N = (int)mxGetScalar(prhs[0]);
    if (Target_N > MAX_N) mexErrMsgIdAndTxt("Heptagon:TargetTooLarge", "Target_N exceeds MAX_N");

    const mxArray* mxE = prhs[1];
    build_encoded_lookup(mxE, (int)mxGetM(mxE), (int)mxGetN(mxE), LOOKUP_E_ENC, true);
    const mxArray* mxA = prhs[2];
    GLOBAL_COLS_A = (int)mxGetN(mxA);
    build_encoded_lookup(mxA, (int)mxGetM(mxA), GLOBAL_COLS_A, LOOKUP_A_ENC, false);

    init_all_lookups(mxA, GLOBAL_COLS_A);

    FIND_ALL_MODE = (mxGetScalar(prhs[3]) > 0.5);
    TARGET_COUNT_LIMIT = (nrhs >= 5) ? (int)mxGetScalar(prhs[4]) : 1;

    SKIP_EDGE_CHECK = (nrhs >= 6) ? (mxGetScalar(prhs[5]) > 0.5) : false;
    SKIP_ANGLE_CHECK = (nrhs >= 7) ? (mxGetScalar(prhs[6]) > 0.5) : false;

    if (SKIP_EDGE_CHECK)  mexPrintf(">> [System Notice] Edge checks are BYPASSED.\n");
    if (SKIP_ANGLE_CHECK) mexPrintf(">> [System Notice] Angle checks are BYPASSED.\n");

    global_solutions.clear();
    stop_search.store(false);
    found_count.store(0);
    global_task_pool.clear();
    idle_threads.store(0);
    omp_init_lock(&pool_lock);

    // =========================================================
    // 【硬件自适应】根据当前 CPU 的物理线程数动态计算最佳种子数量
    // =========================================================
    int num_threads = std::thread::hardware_concurrency(); // 获取系统支持的并发线程数
    if (num_threads < 1) num_threads = 1;

    size_t min_fragments;
    size_t max_fragments;

    if (FIND_ALL_MODE) {
        // 找所有解模式：每个核心至少分配 50 个种子任务，保障深度负载均衡，最低 2000
        min_fragments = std::max((size_t)2000, (size_t)num_threads * 50);
        max_fragments = min_fragments * 6;
    }
    else {
        // 找单个解模式：每个核心分配 8 个种子作为缓冲，既保证多核饱和，又避免过度展开，最低 120
        min_fragments = std::max((size_t)120, (size_t)num_threads * 8);
        max_fragments = min_fragments * 4;
    }

    mexPrintf(">> [Phase 1] Hardware-Adaptive Splitting (Threads: %d, Target Seeds: %d)...\n",
        num_threads, (int)min_fragments);

    GlueState initial_state;
    initial_state.block_count = 1;
    global_task_pool.push_back(initial_state);

    int current_depth = 1;
    while (global_task_pool.size() < min_fragments && current_depth < Target_N) {
        if (stop_search.load(std::memory_order_relaxed)) break;
        std::vector<GlueState> next_pool;

#pragma omp parallel
        {
            std::vector<GlueState> local_next;
            local_next.reserve(1000);
            std::vector<std::pair<EdgeInfo, EdgeInfo>> local_queue;
            local_queue.reserve(64);

#pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < (int)global_task_pool.size(); ++i) {
                if (stop_search.load(std::memory_order_relaxed)) continue;
                const GlueState& S = global_task_pool[i];

                Candidate cand = get_best_candidate(S, local_queue);
                if (cand.is_dead_end || cand.p == -1) continue;

                int p = cand.p; int m = cand.m; int new_id = S.block_count;
                for (int b = 0; b < cand.num_valid_branches; ++b) {
                    int n = cand.valid_branches[b];

                    GlueState next_S;
                    copy_state(next_S, S);
                    next_S.block_count++;
                    next_S.block_type_cw[new_id] = (n >= 7);
                    int connect_side = (n < 7) ? n : n - 7;
                    local_queue.clear();
                    local_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

                    if (perform_zip_and_check(next_S, local_queue)) {
                        if (next_S.block_count == Target_N) record_solution(next_S, Target_N);
                        else local_next.push_back(next_S);
                    }
                }
            }
#pragma omp critical(merge_pool)
            { next_pool.insert(next_pool.end(), local_next.begin(), local_next.end()); }
        }
        global_task_pool = std::move(next_pool);
        current_depth++;
        if (global_task_pool.size() > max_fragments) break;
    }

    if (!FIND_ALL_MODE && !global_task_pool.empty()) {
        std::mt19937 g(static_cast<unsigned int>(time(NULL)));
        std::shuffle(global_task_pool.begin(), global_task_pool.end(), g);
    }

    mexPrintf(">> Generated %d seeds. [Phase 2] Dynamic Work-Stealing Activated!\n", (int)global_task_pool.size());

    // =========================================================
    // Phase 2: 动态多核窃取引擎
    // =========================================================
#pragma omp parallel
    {
#pragma omp single
        { num_total_threads = omp_get_num_threads(); }

        std::vector<GlueState> thread_path(Target_N + 5);
        std::vector<std::pair<EdgeInfo, EdgeInfo>> thread_zip_queue;
        thread_zip_queue.reserve(64);
        uint32_t thread_rng = 123456789 ^ omp_get_thread_num() ^ (unsigned int)time(NULL);

        while (!stop_search.load(std::memory_order_relaxed)) {
            GlueState task;
            bool got_task = false;

            omp_set_lock(&pool_lock);
            if (!global_task_pool.empty()) {
                task = global_task_pool.back();
                global_task_pool.pop_back();
                got_task = true;
            }
            omp_unset_lock(&pool_lock);

            if (got_task) {
                copy_state(thread_path[task.block_count], task);
                worker_dfs_stealing(task.block_count, thread_path, thread_zip_queue, Target_N, thread_rng);
            }
            else {
                idle_threads.fetch_add(1, std::memory_order_relaxed);

                while (global_task_pool.empty() && !stop_search.load(std::memory_order_relaxed)) {
                    if (idle_threads.load(std::memory_order_relaxed) == num_total_threads) {
                        stop_search.store(true, std::memory_order_relaxed);
                        break;
                    }
                    std::this_thread::yield();
                }

                idle_threads.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    omp_destroy_lock(&pool_lock);

    // =========================================================
    // 返回数据
    // =========================================================
    int count = (int)global_solutions.size();
    const char* fields[] = { "count", "solutions" };
    plhs[0] = mxCreateStructMatrix(1, 1, 2, fields);
    mxSetField(plhs[0], 0, "count", mxCreateDoubleScalar((double)count));

    mxArray* out_cell = mxCreateCellMatrix(1, count);
    for (int i = 0; i < count; ++i) {
        mxArray* m = mxCreateNumericMatrix(Target_N, 7, mxINT32_CLASS, mxREAL);
        int* d = (int*)mxGetData(m);
        for (int k = 0; k < Target_N * 7; ++k) d[k] = global_solutions[i].flat_data[k];
        mxSetCell(out_cell, i, m);
    }
    mxSetField(plhs[0], 0, "solutions", out_cell);
}



//下面两个程序不够严谨

// 优化版多核调度 - 支持手动短路跳过边/角验证
/*#include "mex.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <thread>

// ==========================================================
// 基础配置与全局开关
// ==========================================================
const int MAX_N = 300;
const size_t MIN_FRAGMENTS = 3000;
const size_t MAX_FRAGMENTS = 20000;

bool SKIP_EDGE_CHECK = false;   // 手动控制开关：跳过边验证
bool SKIP_ANGLE_CHECK = false;  // 手动控制开关：跳过角验证

inline int get_real_side(int s) { return s % 7; }
inline int get_prev_side(int s) { return (s < 7) ? (s + 6) % 7 : 7 + (s - 7 + 6) % 7; }
inline int get_next_side(int s) { return (s < 7) ? (s + 1) % 7 : 7 + (s - 7 + 1) % 7; }
inline int get_angle_val(int s1, int s2) {
    int r1 = s1 % 7; int r2 = s2 % 7;
    if ((r1 == 0 && r2 == 6) || (r1 == 6 && r2 == 0)) return 0;
    return std::max(r1, r2);
}

std::vector<uint64_t> LOOKUP_E_ENC;
std::vector<uint64_t> LOOKUP_A_ENC;
int GLOBAL_COLS_A = 8;

inline uint64_t encode_edge_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = code * 3 + (row[i] + 1);
    return code;
}
inline uint64_t encode_angle_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = (code << 3) | (row[i] & 0x7);
    return code;
}

struct EdgeInfo {
    int16_t block_id;
    int8_t side;
};

struct GlueState {
    EdgeInfo adj[MAX_N][7];
    int16_t block_count;
    bool block_type_cw[MAX_N];

    GlueState() {
        block_count = 0;
        std::memset(block_type_cw, 0, MAX_N * sizeof(bool));
        for (int i = 0; i < MAX_N; ++i) {
            for (int j = 0; j < 7; ++j) {
                adj[i][j].block_id = -1;
                adj[i][j].side = -1;
            }
        }
    }
};

struct SolutionData {
    int flat_data[MAX_N * 7];
    SolutionData() { std::memset(flat_data, 0, sizeof(flat_data)); }
};

// ==========================================================
// 全局状态与动态负载均衡控制
// ==========================================================
std::vector<SolutionData> global_solutions;
std::atomic<bool> stop_search(false);
std::atomic<int> found_count(0);
bool FIND_ALL_MODE = false;
int TARGET_COUNT_LIMIT = 1;

std::vector<GlueState> global_task_pool;
omp_lock_t pool_lock;
std::atomic<int> idle_threads(0);
int num_total_threads = 1;

inline uint32_t fast_rand(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

// 边查表短路设计
inline bool check_edge_lookup(int side_p, int side_q) {
    if (SKIP_EDGE_CHECK) return true; // 短路：直接判定合法

    int u = get_real_side(side_p); int v = get_real_side(side_q);
    if (u == v) return true;
    int min_s = (u < v) ? u : v; int max_s = (u < v) ? v : u;
    uint64_t code = 0;
    for (int i = 0; i < 7; ++i) {
        int val = 1;
        if (i == min_s) val = 2; else if (i == max_s) val = 0;
        code = code * 3 + val;
    }
    auto it = std::lower_bound(LOOKUP_E_ENC.begin(), LOOKUP_E_ENC.end(), code);
    return (it != LOOKUP_E_ENC.end() && *it == code);
}

// 角查表短路设计
inline bool check_angle_lookup(int a1, int a2, int a3) {
    if (SKIP_ANGLE_CHECK) return true; // 短路：直接判定合法

    uint64_t code = 0; int8_t temp[32] = { 0 };
    if (a1 < GLOBAL_COLS_A - 1 && a1 < 31) temp[a1]++;
    if (a2 < GLOBAL_COLS_A - 1 && a2 < 31) temp[a2]++;
    if (a3 < GLOBAL_COLS_A - 1 && a3 < 31) temp[a3]++;
    if (GLOBAL_COLS_A >= 8 && 7 < 32) temp[7] = 2;
    for (int i = 0; i < GLOBAL_COLS_A && i < 32; ++i) { code = (code << 3) | (temp[i] & 0x7); }
    auto it = std::lower_bound(LOOKUP_A_ENC.begin(), LOOKUP_A_ENC.end(), code);
    return (it != LOOKUP_A_ENC.end() && *it == code);
}

// 拓扑传导与闭合检查函数
bool perform_zip_and_check(GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& queue) {
    size_t head = 0;
    while (head < queue.size()) {
        auto pair = queue[head++];
        EdgeInfo e1 = pair.first; EdgeInfo e2 = pair.second;
        if (!check_edge_lookup(e1.side, e2.side)) return false;

        int p = e1.block_id; int ps = get_real_side(e1.side);
        int q = e2.block_id; int qs = get_real_side(e2.side);

        if (S.adj[p][ps].block_id != -1) {
            if (S.adj[p][ps].block_id != q) return false;
            continue;
        }
        if (S.adj[q][qs].block_id != -1) {
            if (S.adj[q][qs].block_id != p) return false;
            continue;
        }

        S.adj[p][ps] = { (int16_t)q, (int8_t)e2.side }; S.adj[q][qs] = { (int16_t)p, (int8_t)e1.side };
        bool same_dir = (S.block_type_cw[p] == S.block_type_cw[q]);
        int p_neighbors[2] = { get_prev_side(e1.side), get_next_side(e1.side) };
        int q_neighbors[2];
        if (same_dir) {
            q_neighbors[0] = get_next_side(e2.side); q_neighbors[1] = get_prev_side(e2.side);
        }
        else {
            q_neighbors[0] = get_prev_side(e2.side); q_neighbors[1] = get_next_side(e2.side);
        }

        for (int i = 0; i < 2; ++i) {
            int s_p_check = p_neighbors[i]; int s_q_check = q_neighbors[i];
            EdgeInfo link_p = S.adj[p][get_real_side(s_p_check)];
            EdgeInfo link_q = S.adj[q][get_real_side(s_q_check)];
            int k = -1;

            if (link_p.block_id != -1 && link_q.block_id != -1) {
                if (link_p.block_id != link_q.block_id) return false;
                k = link_p.block_id;
                int a1 = get_angle_val(s_p_check, e1.side); int a2 = get_angle_val(e2.side, s_q_check); int a3 = get_angle_val(link_p.side, link_q.side);
                if (!check_angle_lookup(a1, a2, a3)) return false;
            }
            else if (link_p.block_id != -1 && link_q.block_id == -1) {
                k = link_p.block_id; if (k == q) continue;
                bool same_dir_pk = (S.block_type_cw[p] == S.block_type_cw[k]); bool is_p_next = (s_p_check == get_next_side(e1.side));
                int s_k_target = same_dir_pk ? (is_p_next ? get_next_side(link_p.side) : get_prev_side(link_p.side)) : (is_p_next ? get_prev_side(link_p.side) : get_next_side(link_p.side));
                queue.push_back({ EdgeInfo{ (int16_t)q, (int8_t)s_q_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
            else if (link_q.block_id != -1 && link_p.block_id == -1) {
                k = link_q.block_id; if (k == p) continue;
                bool same_dir_qk = (S.block_type_cw[q] == S.block_type_cw[k]); bool is_q_next = (s_q_check == get_next_side(e2.side));
                int s_k_target = same_dir_qk ? (is_q_next ? get_next_side(link_q.side) : get_prev_side(link_q.side)) : (is_q_next ? get_prev_side(link_q.side) : get_next_side(link_q.side));
                queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)s_p_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
        }
    }
    return true;
}

// ==========================================================
// 严格一圈一圈铺设 + 预测剪枝(MRV) 候选选择器
// ==========================================================
struct Candidate {
    int p;
    int m;
    int score;
    bool is_dead_end;
    int num_valid_branches;
    int valid_branches[14];
};

Candidate get_best_candidate(const GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& temp_queue) {
    Candidate result = { -1, -1, -1, false, 0, {0} };

    int min_i = -1;
    for (int i = 0; i < S.block_count; ++i) {
        bool has_open = false;
        for (int j = 0; j < 7; ++j) {
            if (S.adj[i][j].block_id == -1) {
                has_open = true;
                break;
            }
        }
        if (has_open) {
            min_i = i;
            break;
        }
    }

    if (min_i == -1) return result;

    int max_score = -1;
    std::vector<int> candidates_j;
    candidates_j.reserve(7);

    for (int j = 0; j < 7; ++j) {
        if (S.adj[min_i][j].block_id == -1) {
            int score = 0;
            if (S.adj[min_i][get_real_side(get_prev_side(j))].block_id != -1) score += 10;
            if (S.adj[min_i][get_real_side(get_next_side(j))].block_id != -1) score += 10;

            if (score > max_score) {
                max_score = score;
                candidates_j.clear();
                candidates_j.push_back(j);
            }
            else if (score == max_score) {
                candidates_j.push_back(j);
            }
        }
    }

    int best_j = -1;
    int min_valid_branches = 999;
    int best_branches[14];
    int best_branches_count = 0;

    for (int j : candidates_j) {
        int valid_branches[14];
        int valid_count = 0;

        for (int n = 0; n < 14; ++n) {
            GlueState next_S = S;
            next_S.block_count++;
            int new_id = S.block_count;
            next_S.block_type_cw[new_id] = (n >= 7);
            int connect_side = (n < 7) ? n : n - 7;

            temp_queue.clear();
            temp_queue.push_back({ EdgeInfo{ (int16_t)min_i, (int8_t)j }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

            if (perform_zip_and_check(next_S, temp_queue)) {
                valid_branches[valid_count++] = n;
            }
        }

        if (valid_count == 0) {
            result.is_dead_end = true;
            return result;
        }

        if (valid_count < min_valid_branches) {
            min_valid_branches = valid_count;
            best_j = j;
            best_branches_count = valid_count;
            std::memcpy(best_branches, valid_branches, valid_count * sizeof(int));
        }
    }

    result.p = min_i;
    result.m = best_j;
    result.score = max_score;
    result.num_valid_branches = best_branches_count;
    std::memcpy(result.valid_branches, best_branches, best_branches_count * sizeof(int));

    return result;
}

void record_solution(const GlueState& S, int limit) {
    SolutionData sol;
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < 7; ++j) {
            sol.flat_data[i + j * limit] = (S.adj[i][j].block_id != -1) ? (S.adj[i][j].block_id + 1) * 100 + S.adj[i][j].side : 0;
        }
    }
#pragma omp critical(save_sol)
    {
        if (FIND_ALL_MODE || found_count.load(std::memory_order_relaxed) < TARGET_COUNT_LIMIT) {
            global_solutions.push_back(sol);
            if (!FIND_ALL_MODE && ++found_count >= TARGET_COUNT_LIMIT) {
                stop_search.store(true, std::memory_order_relaxed);
            }
        }
    }
}

// ==========================================================
// Phase 2: 动态任务窃取深度优先引擎
// ==========================================================
void worker_dfs_stealing(int depth, std::vector<GlueState>& path_states, std::vector<std::pair<EdgeInfo, EdgeInfo>>& zip_queue, int limit, uint32_t& rng_state) {
    if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

    const GlueState& S = path_states[depth];
    if (S.block_count == limit) { record_solution(S, limit); return; }

    Candidate cand = get_best_candidate(S, zip_queue);
    if (cand.is_dead_end || cand.p == -1) return;

    bool should_donate = false;
    if (depth < limit - 4 && idle_threads.load(std::memory_order_relaxed) > 0) {
        if (omp_test_lock(&pool_lock)) {
            if (global_task_pool.size() < 10000) { should_donate = true; }
            omp_unset_lock(&pool_lock);
        }
    }

    int p = cand.p; int m = cand.m; int new_id = S.block_count;
    int num_branches = cand.num_valid_branches;
    int branches[14];
    std::memcpy(branches, cand.valid_branches, num_branches * sizeof(int));

    if (!FIND_ALL_MODE) {
        for (int i = num_branches - 1; i > 0; --i) {
            int j = fast_rand(rng_state) % (i + 1); std::swap(branches[i], branches[j]);
        }
    }

    std::vector<GlueState> valid_children;
    if (should_donate) valid_children.reserve(num_branches);

    for (int i = 0; i < num_branches; ++i) {
        if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

        int n = branches[i];
        GlueState next_S = S;
        next_S.block_count++;
        next_S.block_type_cw[new_id] = (n >= 7);
        int connect_side = (n < 7) ? n : n - 7;

        zip_queue.clear();
        zip_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

        if (perform_zip_and_check(next_S, zip_queue)) {
            if (should_donate) {
                valid_children.push_back(next_S);
            }
            else {
                path_states[depth + 1] = next_S;
                worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
            }
        }
    }

    if (should_donate && !valid_children.empty()) {
        omp_set_lock(&pool_lock);
        for (size_t i = 1; i < valid_children.size(); ++i) {
            global_task_pool.push_back(valid_children[i]);
        }
        omp_unset_lock(&pool_lock);

        path_states[depth + 1] = valid_children[0];
        worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
    }
}

void build_encoded_lookup(const mxArray* mxData, int rows, int cols, std::vector<uint64_t>& table, bool is_edge) {
    double* pr = mxGetPr(mxData);
    table.clear(); table.reserve(rows);
    std::vector<int8_t> temp_row(cols);
    for (int r = 0; r < rows; ++r) {
        bool all_zero = true;
        for (int c = 0; c < cols; ++c) {
            temp_row[c] = (int8_t)std::round(pr[r + c * rows]);
            if (temp_row[c] != 0) all_zero = false;
        }
        if (all_zero) continue;
        if (is_edge) {
            int pos_idx = -1, neg_idx = -1;
            for (int c = 0; c < cols; ++c) {
                if (temp_row[c] == 1) pos_idx = c;
                if (temp_row[c] == -1) neg_idx = c;
            }
            if (pos_idx != -1 && neg_idx != -1) {
                std::fill(temp_row.begin(), temp_row.end(), 0);
                temp_row[std::min(pos_idx, neg_idx)] = 1; temp_row[std::max(pos_idx, neg_idx)] = -1;
            }
            table.push_back(encode_edge_vec(temp_row.data(), cols));
        }
        else {
            table.push_back(encode_angle_vec(temp_row.data(), cols));
        }
    }
    std::sort(table.begin(), table.end());
    table.erase(std::unique(table.begin(), table.end()), table.end());
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // 基础输入检查
    if (nrhs < 4) mexErrMsgTxt("Usage: heptagon_search(...)");

    int Target_N = (int)mxGetScalar(prhs[0]);
    if (Target_N > MAX_N) mexErrMsgIdAndTxt("Heptagon:TargetTooLarge", "Target_N exceeds MAX_N");

    const mxArray* mxE = prhs[1];
    build_encoded_lookup(mxE, (int)mxGetM(mxE), (int)mxGetN(mxE), LOOKUP_E_ENC, true);
    const mxArray* mxA = prhs[2];
    GLOBAL_COLS_A = (int)mxGetN(mxA);
    build_encoded_lookup(mxA, (int)mxGetM(mxA), GLOBAL_COLS_A, LOOKUP_A_ENC, false);

    FIND_ALL_MODE = (mxGetScalar(prhs[3]) > 0.5);
    TARGET_COUNT_LIMIT = (nrhs >= 5) ? (int)mxGetScalar(prhs[4]) : 1;

    // 获取第 6、7 个参数（若不传入，默认不跳过，值为 0）
    SKIP_EDGE_CHECK = (nrhs >= 6) ? (mxGetScalar(prhs[5]) > 0.5) : false;
    SKIP_ANGLE_CHECK = (nrhs >= 7) ? (mxGetScalar(prhs[6]) > 0.5) : false;

    if (SKIP_EDGE_CHECK)  mexPrintf(">> [System Notice] Edge checks are BYPASSED.\n");
    if (SKIP_ANGLE_CHECK) mexPrintf(">> [System Notice] Angle checks are BYPASSED.\n");

    global_solutions.clear();
    stop_search.store(false);
    found_count.store(0);
    global_task_pool.clear();
    idle_threads.store(0);
    omp_init_lock(&pool_lock);

    // =========================================================
    // Phase 1: 基础广度展开
    // =========================================================
    mexPrintf(">> [Phase 1] Adaptive Breadth Splitting...\n");

    GlueState initial_state;
    initial_state.block_count = 1;
    global_task_pool.push_back(initial_state);

    int current_depth = 1;
    while (global_task_pool.size() < MIN_FRAGMENTS && current_depth < Target_N) {
        if (stop_search.load(std::memory_order_relaxed)) break;
        std::vector<GlueState> next_pool;

#pragma omp parallel
        {
            std::vector<GlueState> local_next;
            local_next.reserve(1000);
            std::vector<std::pair<EdgeInfo, EdgeInfo>> local_queue;
            local_queue.reserve(64);

#pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < (int)global_task_pool.size(); ++i) {
                if (stop_search.load(std::memory_order_relaxed)) continue;
                const GlueState& S = global_task_pool[i];

                Candidate cand = get_best_candidate(S, local_queue);
                if (cand.is_dead_end || cand.p == -1) continue;

                int p = cand.p; int m = cand.m; int new_id = S.block_count;
                for (int b = 0; b < cand.num_valid_branches; ++b) {
                    int n = cand.valid_branches[b];
                    GlueState next_S = S;
                    next_S.block_count++;
                    next_S.block_type_cw[new_id] = (n >= 7);
                    int connect_side = (n < 7) ? n : n - 7;
                    local_queue.clear();
                    local_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

                    if (perform_zip_and_check(next_S, local_queue)) {
                        if (next_S.block_count == Target_N) record_solution(next_S, Target_N);
                        else local_next.push_back(next_S);
                    }
                }
            }
#pragma omp critical(merge_pool)
            { next_pool.insert(next_pool.end(), local_next.begin(), local_next.end()); }
        }
        global_task_pool = std::move(next_pool);
        current_depth++;
        if (global_task_pool.size() > MAX_FRAGMENTS) break;
    }

    if (!FIND_ALL_MODE && !global_task_pool.empty()) {
        srand((unsigned int)time(NULL));
        std::random_shuffle(global_task_pool.begin(), global_task_pool.end());
    }
    mexPrintf(">> Generated %d seeds. [Phase 2] Dynamic Work-Stealing Activated!\n", global_task_pool.size());

    // =========================================================
    // Phase 2: 动态多核窃取引擎
    // =========================================================
#pragma omp parallel
    {
#pragma omp single
        { num_total_threads = omp_get_num_threads(); }

        std::vector<GlueState> thread_path(Target_N + 5);
        std::vector<std::pair<EdgeInfo, EdgeInfo>> thread_zip_queue;
        thread_zip_queue.reserve(64);
        uint32_t thread_rng = 123456789 ^ omp_get_thread_num() ^ (unsigned int)time(NULL);

        while (!stop_search.load(std::memory_order_relaxed)) {
            GlueState task;
            bool got_task = false;

            omp_set_lock(&pool_lock);
            if (!global_task_pool.empty()) {
                task = global_task_pool.back();
                global_task_pool.pop_back();
                got_task = true;
            }
            omp_unset_lock(&pool_lock);

            if (got_task) {
                thread_path[task.block_count] = task;
                worker_dfs_stealing(task.block_count, thread_path, thread_zip_queue, Target_N, thread_rng);
            }
            else {
                idle_threads.fetch_add(1, std::memory_order_relaxed);

                while (global_task_pool.empty() && !stop_search.load(std::memory_order_relaxed)) {
                    if (idle_threads.load(std::memory_order_relaxed) == num_total_threads) {
                        stop_search.store(true, std::memory_order_relaxed);
                        break;
                    }
                    std::this_thread::yield();
                }

                idle_threads.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    omp_destroy_lock(&pool_lock);

    // =========================================================
    // 返回数据
    // =========================================================
    int count = (int)global_solutions.size();
    const char* fields[] = { "count", "solutions" };
    plhs[0] = mxCreateStructMatrix(1, 1, 2, fields);
    mxSetField(plhs[0], 0, "count", mxCreateDoubleScalar((double)count));

    mxArray* out_cell = mxCreateCellMatrix(1, count);
    for (int i = 0; i < count; ++i) {
        mxArray* m = mxCreateNumericMatrix(Target_N, 7, mxINT32_CLASS, mxREAL);
        int* d = (int*)mxGetData(m);
        for (int k = 0; k < Target_N * 7; ++k) d[k] = global_solutions[i].flat_data[k];
        mxSetCell(out_cell, i, m);
    }
    mxSetField(plhs[0], 0, "solutions", out_cell);
}*/


// 优化版多核调度 - 支持手动短路跳过边/角验证 + 2阶顶点子集预判剪枝优化 (去通信死锁稳定版)
/*#include "mex.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <thread>
#include <random>
#include <ctime>

// ==========================================================
// 基础配置与全局开关
// ==========================================================
const int MAX_N = 1000;

bool SKIP_EDGE_CHECK = false;   // 手动控制开关：跳过边验证
bool SKIP_ANGLE_CHECK = false;  // 手动控制开关：跳过角验证

inline int get_real_side(int s) { return s % 7; }
inline int get_prev_side(int s) { return (s < 7) ? (s + 6) % 7 : 7 + (s - 7 + 6) % 7; }
inline int get_next_side(int s) { return (s < 7) ? (s + 1) % 7 : 7 + (s - 7 + 1) % 7; }
inline int get_angle_val(int s1, int s2) {
    int r1 = s1 % 7; int r2 = s2 % 7;
    if ((r1 == 0 && r2 == 6) || (r1 == 6 && r2 == 0)) return 0;
    return std::max(r1, r2);
}

std::vector<uint64_t> LOOKUP_E_ENC;
std::vector<uint64_t> LOOKUP_A_ENC;
int GLOBAL_COLS_A = 8;

// ==========================================================
// 【优化】超快 O(1) 全局查找表，免去运行时二分查找与编码开销
// ==========================================================
bool VALID_2_DEGREE[8][8] = { false };
bool VALID_3_DEGREE[8][8][8] = { false };
bool VALID_EDGE_TRANSITION[7][7] = { false };

struct EdgeInfo {
    int16_t block_id;
    int8_t side;
};

struct GlueState {
    EdgeInfo adj[MAX_N][7];
    int16_t block_count;
    bool block_type_cw[MAX_N];

    GlueState() {
        block_count = 0;
        std::memset(block_type_cw, 0, MAX_N * sizeof(bool));
        for (int i = 0; i < MAX_N; ++i) {
            for (int j = 0; j < 7; ++j) {
                adj[i][j].block_id = -1;
                adj[i][j].side = -1;
            }
        }
    }
};

// 增量状态复制
inline void copy_state(GlueState& dest, const GlueState& src) {
    dest.block_count = src.block_count;
    std::memcpy(dest.adj, src.adj, src.block_count * sizeof(src.adj[0]));
    std::memcpy(dest.block_type_cw, src.block_type_cw, src.block_count * sizeof(bool));

    for (int j = 0; j < 7; ++j) {
        dest.adj[src.block_count][j].block_id = -1;
        dest.adj[src.block_count][j].side = -1;
    }
}

inline uint64_t encode_edge_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = code * 3 + (row[i] + 1);
    return code;
}
inline uint64_t encode_angle_vec(const int8_t* row, int cols) {
    uint64_t code = 0;
    for (int i = 0; i < cols; ++i) code = (code << 3) | (row[i] & 0x7);
    return code;
}

// 初始化全局高速过滤表
void init_all_lookups(const mxArray* mxA, int cols_A) {
    int rows_A = (int)mxGetM(mxA);
    double* pr_A = mxGetPr(mxA);

    std::memset(VALID_2_DEGREE, 0, sizeof(VALID_2_DEGREE));
    for (int r = 0; r < rows_A; ++r) {
        int counts[7] = { 0 };
        for (int c = 0; c < 7 && c < cols_A - 1; ++c) {
            counts[c] = (int)std::round(pr_A[r + c * rows_A]);
        }
        std::vector<int> angles;
        for (int i = 0; i < 7; ++i) {
            for (int k = 0; k < counts[i]; ++k) {
                angles.push_back(i);
            }
        }
        if (angles.size() == 3) {
            int a = angles[0]; int b = angles[1]; int c = angles[2];
            if (a < 8 && b < 8 && c < 8) {
                VALID_2_DEGREE[a][b] = true; VALID_2_DEGREE[b][a] = true;
                VALID_2_DEGREE[b][c] = true; VALID_2_DEGREE[c][b] = true;
                VALID_2_DEGREE[a][c] = true; VALID_2_DEGREE[c][a] = true;
            }
        }
    }

    std::memset(VALID_3_DEGREE, 0, sizeof(VALID_3_DEGREE));
    for (int a1 = 0; a1 < 8; ++a1) {
        for (int a2 = 0; a2 < 8; ++a2) {
            for (int a3 = 0; a3 < 8; ++a3) {
                uint64_t code = 0; int8_t temp[32] = { 0 };
                if (a1 < cols_A - 1) temp[a1]++;
                if (a2 < cols_A - 1) temp[a2]++;
                if (a3 < cols_A - 1) temp[a3]++;
                if (cols_A >= 8) temp[7] = 2;
                for (int i = 0; i < cols_A; ++i) { code = (code << 3) | (temp[i] & 0x7); }
                auto it = std::lower_bound(LOOKUP_A_ENC.begin(), LOOKUP_A_ENC.end(), code);
                if (it != LOOKUP_A_ENC.end() && *it == code) {
                    VALID_3_DEGREE[a1][a2][a3] = true;
                }
            }
        }
    }

    std::memset(VALID_EDGE_TRANSITION, 0, sizeof(VALID_EDGE_TRANSITION));
    for (int u = 0; u < 7; ++u) {
        for (int v = 0; v < 7; ++v) {
            if (u == v) {
                VALID_EDGE_TRANSITION[u][v] = true;
            }
            else {
                int min_s = std::min(u, v); int max_s = std::max(u, v);
                uint64_t code = 0;
                for (int i = 0; i < 7; ++i) {
                    int val = 1;
                    if (i == min_s) val = 2; else if (i == max_s) val = 0;
                    code = code * 3 + val;
                }
                auto it = std::lower_bound(LOOKUP_E_ENC.begin(), LOOKUP_E_ENC.end(), code);
                VALID_EDGE_TRANSITION[u][v] = (it != LOOKUP_E_ENC.end() && *it == code);
            }
        }
    }
}

struct SolutionData {
    int flat_data[MAX_N * 7];
    SolutionData() { std::memset(flat_data, 0, sizeof(flat_data)); }
};

// ==========================================================
// 全局状态与动态负载均衡控制
// ==========================================================
std::vector<SolutionData> global_solutions;
std::atomic<bool> stop_search(false);
std::atomic<int> found_count(0);
bool FIND_ALL_MODE = false;
int TARGET_COUNT_LIMIT = 1;

std::vector<GlueState> global_task_pool;
omp_lock_t pool_lock;
std::atomic<int> idle_threads(0);
int num_total_threads = 1;

inline uint32_t fast_rand(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

inline bool check_edge_lookup(int side_p, int side_q) {
    if (SKIP_EDGE_CHECK) return true;
    return VALID_EDGE_TRANSITION[get_real_side(side_p)][get_real_side(side_q)];
}

inline bool check_angle_lookup(int a1, int a2, int a3) {
    if (SKIP_ANGLE_CHECK) return true;
    if (a1 >= 8 || a2 >= 8 || a3 >= 8 || a1 < 0 || a2 < 0 || a3 < 0) return false;
    return VALID_3_DEGREE[a1][a2][a3];
}

// 拓扑传导与闭合检查
bool perform_zip_and_check(GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& queue) {
    size_t head = 0;
    while (head < queue.size()) {
        auto pair = queue[head++];
        EdgeInfo e1 = pair.first; EdgeInfo e2 = pair.second;
        if (!check_edge_lookup(e1.side, e2.side)) return false;

        int p = e1.block_id; int ps = get_real_side(e1.side);
        int q = e2.block_id; int qs = get_real_side(e2.side);

        if (S.adj[p][ps].block_id != -1) {
            if (S.adj[p][ps].block_id != q) return false;
            continue;
        }
        if (S.adj[q][qs].block_id != -1) {
            if (S.adj[q][qs].block_id != p) return false;
            continue;
        }

        S.adj[p][ps] = { (int16_t)q, (int8_t)e2.side }; S.adj[q][qs] = { (int16_t)p, (int8_t)e1.side };
        bool same_dir = (S.block_type_cw[p] == S.block_type_cw[q]);
        int p_neighbors[2] = { get_prev_side(e1.side), get_next_side(e1.side) };
        int q_neighbors[2];
        if (same_dir) {
            q_neighbors[0] = get_next_side(e2.side); q_neighbors[1] = get_prev_side(e2.side);
        }
        else {
            q_neighbors[0] = get_prev_side(e2.side); q_neighbors[1] = get_next_side(e2.side);
        }

        for (int i = 0; i < 2; ++i) {
            int s_p_check = p_neighbors[i]; int s_q_check = q_neighbors[i];

            if (!SKIP_ANGLE_CHECK) {
                int a1 = get_angle_val(s_p_check, e1.side);
                int a2 = get_angle_val(e2.side, s_q_check);
                if (a1 < 8 && a2 < 8 && !VALID_2_DEGREE[a1][a2]) {
                    return false;
                }
            }

            EdgeInfo link_p = S.adj[p][get_real_side(s_p_check)];
            EdgeInfo link_q = S.adj[q][get_real_side(s_q_check)];
            int k = -1;

            if (link_p.block_id != -1 && link_q.block_id != -1) {
                if (link_p.block_id != link_q.block_id) return false;
                k = link_p.block_id;
                int a1 = get_angle_val(s_p_check, e1.side); int a2 = get_angle_val(e2.side, s_q_check); int a3 = get_angle_val(link_p.side, link_q.side);
                if (!check_angle_lookup(a1, a2, a3)) return false;
            }
            else if (link_p.block_id != -1 && link_q.block_id == -1) {
                k = link_p.block_id; if (k == q) continue;
                bool same_dir_pk = (S.block_type_cw[p] == S.block_type_cw[k]); bool is_p_next = (s_p_check == get_next_side(e1.side));
                int s_k_target = same_dir_pk ? (is_p_next ? get_next_side(link_p.side) : get_prev_side(link_p.side)) : (is_p_next ? get_prev_side(link_p.side) : get_next_side(link_p.side));
                queue.push_back({ EdgeInfo{ (int16_t)q, (int8_t)s_q_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
            else if (link_q.block_id != -1 && link_p.block_id == -1) {
                k = link_q.block_id; if (k == p) continue;
                bool same_dir_qk = (S.block_type_cw[q] == S.block_type_cw[k]); bool is_q_next = (s_q_check == get_next_side(e2.side));
                int s_k_target = same_dir_qk ? (is_q_next ? get_next_side(link_q.side) : get_prev_side(link_q.side)) : (is_q_next ? get_prev_side(link_q.side) : get_next_side(link_q.side));
                queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)s_p_check }, EdgeInfo{ (int16_t)k, (int8_t)s_k_target } });
            }
        }
    }
    return true;
}

// MRV 候选选择器
struct Candidate {
    int p;
    int m;
    int score;
    bool is_dead_end;
    int num_valid_branches;
    int valid_branches[14];
};

Candidate get_best_candidate(const GlueState& S, std::vector<std::pair<EdgeInfo, EdgeInfo>>& temp_queue) {
    Candidate result = { -1, -1, -1, false, 0, {0} };

    int min_i = -1;
    for (int i = 0; i < S.block_count; ++i) {
        bool has_open = false;
        for (int j = 0; j < 7; ++j) {
            if (S.adj[i][j].block_id == -1) {
                has_open = true;
                break;
            }
        }
        if (has_open) {
            min_i = i;
            break;
        }
    }

    if (min_i == -1) return result;

    int max_score = -1;
    std::vector<int> candidates_j;
    candidates_j.reserve(7);

    for (int j = 0; j < 7; ++j) {
        if (S.adj[min_i][j].block_id == -1) {
            int score = 0;
            if (S.adj[min_i][get_real_side(get_prev_side(j))].block_id != -1) score += 10;
            if (S.adj[min_i][get_real_side(get_next_side(j))].block_id != -1) score += 10;

            if (score > max_score) {
                max_score = score;
                candidates_j.clear();
                candidates_j.push_back(j);
            }
            else if (score == max_score) {
                candidates_j.push_back(j);
            }
        }
    }

    int best_j = -1;
    int min_valid_branches = 999;
    int best_branches[14];
    int best_branches_count = 0;

    GlueState next_S;

    for (int j : candidates_j) {
        int valid_branches[14];
        int valid_count = 0;

        for (int n = 0; n < 14; ++n) {
            copy_state(next_S, S);
            next_S.block_count++;
            int new_id = S.block_count;
            next_S.block_type_cw[new_id] = (n >= 7);
            int connect_side = (n < 7) ? n : n - 7;

            temp_queue.clear();
            temp_queue.push_back({ EdgeInfo{ (int16_t)min_i, (int8_t)j }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

            if (perform_zip_and_check(next_S, temp_queue)) {
                valid_branches[valid_count++] = n;
            }
        }

        if (valid_count == 0) {
            result.is_dead_end = true;
            return result;
        }

        if (valid_count < min_valid_branches) {
            min_valid_branches = valid_count;
            best_j = j;
            best_branches_count = valid_count;
            std::memcpy(best_branches, valid_branches, valid_count * sizeof(int));
        }
    }

    result.p = min_i;
    result.m = best_j;
    result.score = max_score;
    result.num_valid_branches = best_branches_count;
    std::memcpy(result.valid_branches, best_branches, best_branches_count * sizeof(int));

    return result;
}

void record_solution(const GlueState& S, int limit) {
    SolutionData sol;
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < 7; ++j) {
            sol.flat_data[i + j * limit] = (S.adj[i][j].block_id != -1) ? (S.adj[i][j].block_id + 1) * 100 + S.adj[i][j].side : 0;
        }
    }
#pragma omp critical(save_sol)
    {
        if (FIND_ALL_MODE || found_count.load(std::memory_order_relaxed) < TARGET_COUNT_LIMIT) {
            global_solutions.push_back(sol);
            if (!FIND_ALL_MODE && ++found_count >= TARGET_COUNT_LIMIT) {
                stop_search.store(true, std::memory_order_relaxed);
            }
        }
    }
}

// Phase 2: 动态任务窃取深度优先引擎 (去交互死锁版)
void worker_dfs_stealing(int depth, std::vector<GlueState>& path_states, std::vector<std::pair<EdgeInfo, EdgeInfo>>& zip_queue, int limit, uint32_t& rng_state) {
    if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

    const GlueState& S = path_states[depth];
    if (S.block_count == limit) { record_solution(S, limit); return; }

    Candidate cand = get_best_candidate(S, zip_queue);
    if (cand.is_dead_end || cand.p == -1) return;

    bool should_donate = false;
    if (depth < limit - 4 && idle_threads.load(std::memory_order_relaxed) > 0) {
        if (omp_test_lock(&pool_lock)) {
            if (global_task_pool.size() < 10000) { should_donate = true; }
            omp_unset_lock(&pool_lock);
        }
    }

    int p = cand.p; int m = cand.m; int new_id = S.block_count;
    int num_branches = cand.num_valid_branches;
    int branches[14];
    std::memcpy(branches, cand.valid_branches, num_branches * sizeof(int));

    if (!FIND_ALL_MODE) {
        for (int i = num_branches - 1; i > 0; --i) {
            int j = fast_rand(rng_state) % (i + 1); std::swap(branches[i], branches[j]);
        }
    }

    std::vector<GlueState> valid_children;
    if (should_donate) valid_children.reserve(num_branches);

    for (int i = 0; i < num_branches; ++i) {
        if (!FIND_ALL_MODE && stop_search.load(std::memory_order_relaxed)) return;

        int n = branches[i];

        copy_state(path_states[depth + 1], S);
        GlueState& next_S = path_states[depth + 1];
        next_S.block_count++;
        next_S.block_type_cw[new_id] = (n >= 7);
        int connect_side = (n < 7) ? n : n - 7;

        zip_queue.clear();
        zip_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

        if (perform_zip_and_check(next_S, zip_queue)) {
            if (should_donate) {
                valid_children.push_back(next_S);
            }
            else {
                worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
            }
        }
    }

    if (should_donate && !valid_children.empty()) {
        omp_set_lock(&pool_lock);
        for (size_t i = 1; i < valid_children.size(); ++i) {
            global_task_pool.push_back(valid_children[i]);
        }
        omp_unset_lock(&pool_lock);

        copy_state(path_states[depth + 1], valid_children[0]);
        worker_dfs_stealing(depth + 1, path_states, zip_queue, limit, rng_state);
    }
}

void build_encoded_lookup(const mxArray* mxData, int rows, int cols, std::vector<uint64_t>& table, bool is_edge) {
    double* pr = mxGetPr(mxData);
    table.clear(); table.reserve(rows);
    std::vector<int8_t> temp_row(cols);
    for (int r = 0; r < rows; ++r) {
        bool all_zero = true;
        for (int c = 0; c < cols; ++c) {
            temp_row[c] = (int8_t)std::round(pr[r + c * rows]);
            if (temp_row[c] != 0) all_zero = false;
        }
        if (all_zero) continue;
        if (is_edge) {
            int pos_idx = -1, neg_idx = -1;
            for (int c = 0; c < cols; ++c) {
                if (temp_row[c] == 1) pos_idx = c;
                if (temp_row[c] == -1) neg_idx = c;
            }
            if (pos_idx != -1 && neg_idx != -1) {
                std::fill(temp_row.begin(), temp_row.end(), 0);
                temp_row[std::min(pos_idx, neg_idx)] = 1; temp_row[std::max(pos_idx, neg_idx)] = -1;
            }
            table.push_back(encode_edge_vec(temp_row.data(), cols));
        }
        else {
            table.push_back(encode_angle_vec(temp_row.data(), cols));
        }
    }
    std::sort(table.begin(), table.end());
    table.erase(std::unique(table.begin(), table.end()), table.end());
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 4) mexErrMsgTxt("Usage: heptagon_search(...)");

    int Target_N = (int)mxGetScalar(prhs[0]);
    if (Target_N > MAX_N) mexErrMsgIdAndTxt("Heptagon:TargetTooLarge", "Target_N exceeds MAX_N");

    const mxArray* mxE = prhs[1];
    build_encoded_lookup(mxE, (int)mxGetM(mxE), (int)mxGetN(mxE), LOOKUP_E_ENC, true);
    const mxArray* mxA = prhs[2];
    GLOBAL_COLS_A = (int)mxGetN(mxA);
    build_encoded_lookup(mxA, (int)mxGetM(mxA), GLOBAL_COLS_A, LOOKUP_A_ENC, false);

    init_all_lookups(mxA, GLOBAL_COLS_A);

    FIND_ALL_MODE = (mxGetScalar(prhs[3]) > 0.5);
    TARGET_COUNT_LIMIT = (nrhs >= 5) ? (int)mxGetScalar(prhs[4]) : 1;

    SKIP_EDGE_CHECK = (nrhs >= 6) ? (mxGetScalar(prhs[5]) > 0.5) : false;
    SKIP_ANGLE_CHECK = (nrhs >= 7) ? (mxGetScalar(prhs[6]) > 0.5) : false;

    if (SKIP_EDGE_CHECK)  mexPrintf(">> [System Notice] Edge checks are BYPASSED.\n");
    if (SKIP_ANGLE_CHECK) mexPrintf(">> [System Notice] Angle checks are BYPASSED.\n");

    global_solutions.clear();
    stop_search.store(false);
    found_count.store(0);
    global_task_pool.clear();
    idle_threads.store(0);
    omp_init_lock(&pool_lock);

    // =========================================================
    // 【硬件自适应】根据当前 CPU 的物理线程数动态计算最佳种子数量
    // =========================================================
    int num_threads = std::thread::hardware_concurrency(); // 获取系统支持的并发线程数
    if (num_threads < 1) num_threads = 1;

    size_t min_fragments;
    size_t max_fragments;

    if (FIND_ALL_MODE) {
        // 找所有解模式：每个核心至少分配 50 个种子任务，保障深度负载均衡，最低 2000
        min_fragments = std::max((size_t)2000, (size_t)num_threads * 50);
        max_fragments = min_fragments * 6;
    }
    else {
        // 找单个解模式：每个核心分配 8 个种子作为缓冲，既保证多核饱和，又避免过度展开，最低 120
        min_fragments = std::max((size_t)120, (size_t)num_threads * 8);
        max_fragments = min_fragments * 4;
    }

    mexPrintf(">> [Phase 1] Hardware-Adaptive Splitting (Threads: %d, Target Seeds: %d)...\n",
        num_threads, (int)min_fragments);

    GlueState initial_state;
    initial_state.block_count = 1;
    global_task_pool.push_back(initial_state);

    int current_depth = 1;
    while (global_task_pool.size() < min_fragments && current_depth < Target_N) {
        if (stop_search.load(std::memory_order_relaxed)) break;
        std::vector<GlueState> next_pool;

#pragma omp parallel
        {
            std::vector<GlueState> local_next;
            local_next.reserve(1000);
            std::vector<std::pair<EdgeInfo, EdgeInfo>> local_queue;
            local_queue.reserve(64);

#pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < (int)global_task_pool.size(); ++i) {
                if (stop_search.load(std::memory_order_relaxed)) continue;
                const GlueState& S = global_task_pool[i];

                Candidate cand = get_best_candidate(S, local_queue);
                if (cand.is_dead_end || cand.p == -1) continue;

                int p = cand.p; int m = cand.m; int new_id = S.block_count;
                for (int b = 0; b < cand.num_valid_branches; ++b) {
                    int n = cand.valid_branches[b];

                    GlueState next_S;
                    copy_state(next_S, S);
                    next_S.block_count++;
                    next_S.block_type_cw[new_id] = (n >= 7);
                    int connect_side = (n < 7) ? n : n - 7;
                    local_queue.clear();
                    local_queue.push_back({ EdgeInfo{ (int16_t)p, (int8_t)m }, EdgeInfo{ (int16_t)new_id, (int8_t)connect_side } });

                    if (perform_zip_and_check(next_S, local_queue)) {
                        if (next_S.block_count == Target_N) record_solution(next_S, Target_N);
                        else local_next.push_back(next_S);
                    }
                }
            }
#pragma omp critical(merge_pool)
            { next_pool.insert(next_pool.end(), local_next.begin(), local_next.end()); }
        }
        global_task_pool = std::move(next_pool);
        current_depth++;
        if (global_task_pool.size() > max_fragments) break;
    }

    if (!FIND_ALL_MODE && !global_task_pool.empty()) {
        std::mt19937 g(static_cast<unsigned int>(time(NULL)));
        std::shuffle(global_task_pool.begin(), global_task_pool.end(), g);
    }

    mexPrintf(">> Generated %d seeds. [Phase 2] Dynamic Work-Stealing Activated!\n", (int)global_task_pool.size());

    // =========================================================
    // Phase 2: 动态多核窃取引擎
    // =========================================================
#pragma omp parallel
    {
#pragma omp single
        { num_total_threads = omp_get_num_threads(); }

        std::vector<GlueState> thread_path(Target_N + 5);
        std::vector<std::pair<EdgeInfo, EdgeInfo>> thread_zip_queue;
        thread_zip_queue.reserve(64);
        uint32_t thread_rng = 123456789 ^ omp_get_thread_num() ^ (unsigned int)time(NULL);

        while (!stop_search.load(std::memory_order_relaxed)) {
            GlueState task;
            bool got_task = false;

            omp_set_lock(&pool_lock);
            if (!global_task_pool.empty()) {
                task = global_task_pool.back();
                global_task_pool.pop_back();
                got_task = true;
            }
            omp_unset_lock(&pool_lock);

            if (got_task) {
                copy_state(thread_path[task.block_count], task);
                worker_dfs_stealing(task.block_count, thread_path, thread_zip_queue, Target_N, thread_rng);
            }
            else {
                idle_threads.fetch_add(1, std::memory_order_relaxed);

                while (global_task_pool.empty() && !stop_search.load(std::memory_order_relaxed)) {
                    if (idle_threads.load(std::memory_order_relaxed) == num_total_threads) {
                        stop_search.store(true, std::memory_order_relaxed);
                        break;
                    }
                    std::this_thread::yield();
                }

                idle_threads.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    omp_destroy_lock(&pool_lock);

    // =========================================================
    // 返回数据
    // =========================================================
    int count = (int)global_solutions.size();
    const char* fields[] = { "count", "solutions" };
    plhs[0] = mxCreateStructMatrix(1, 1, 2, fields);
    mxSetField(plhs[0], 0, "count", mxCreateDoubleScalar((double)count));

    mxArray* out_cell = mxCreateCellMatrix(1, count);
    for (int i = 0; i < count; ++i) {
        mxArray* m = mxCreateNumericMatrix(Target_N, 7, mxINT32_CLASS, mxREAL);
        int* d = (int*)mxGetData(m);
        for (int k = 0; k < Target_N * 7; ++k) d[k] = global_solutions[i].flat_data[k];
        mxSetCell(out_cell, i, m);
    }
    mxSetField(plhs[0], 0, "solutions", out_cell);
}*/










