// pingheng: 0 = no balance, 1 = legacy weak balance,
//            2 = weak balance plus completed-space strong balance.
// Strong balance is applied whenever Target_RA > 0 and never to edge-only data.

#include "mex.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <atomic>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

    constexpr std::int64_t PRIME1 = 1000000007LL;
    constexpr std::int64_t PRIME2 = 1000000009LL;

    struct Mat {
        int rows = 0;
        int cols = 0;
        std::vector<int> x;

        Mat() = default;
        Mat(int r, int c) : rows(r), cols(c), x(static_cast<std::size_t>(r)* c, 0) {}

        int& at(int r, int c) {
            return x[static_cast<std::size_t>(r) * cols + c];
        }

        int at(int r, int c) const {
            return x[static_cast<std::size_t>(r) * cols + c];
        }
    };

    struct Candidate {
        Mat mat;
        std::string key;
    };

    struct PairCandidate {
        Mat edge;
        Mat angle;
        std::string key;
    };

    struct KnownPair {
        Mat edge;
        Mat angle;
    };

    enum class AngleKeyMode {
        Exact,
        Symmetric,
        Dihedral
    };

    int choose_workers(int requested, long long work_items) {
        if (work_items <= 1) {
            return 1;
        }
        const long long capped = std::min<long long>(std::max(1, requested), work_items);
        return static_cast<int>(capped);
    }

    template <class Fn>
    void parallel_for_dynamic(long long begin, long long end, int workers, Fn fn) {
        if (workers <= 1 || end <= begin + 1) {
            for (long long i = begin; i < end; ++i) {
                fn(0, i);
            }
            return;
        }

        std::atomic<long long> next(begin);
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(workers));

        for (int tid = 0; tid < workers; ++tid) {
            threads.emplace_back([&, tid]() {
                while (true) {
                    const long long i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= end) {
                        break;
                    }
                    fn(tid, i);
                }
                });
        }

        for (std::thread& t : threads) {
            t.join();
        }
    }

    std::int64_t mod_norm(std::int64_t v, std::int64_t p) {
        v %= p;
        return v < 0 ? v + p : v;
    }

    std::int64_t mod_pow(std::int64_t a, std::int64_t e, std::int64_t p) {
        std::int64_t r = 1;
        a = mod_norm(a, p);
        while (e > 0) {
            if (e & 1) {
                r = (r * a) % p;
            }
            a = (a * a) % p;
            e >>= 1;
        }
        return r;
    }

    void append_u32(std::string& s, std::uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            s.push_back(static_cast<char>((v >> (8 * i)) & 0xffu));
        }
    }

    int rank_mod(const Mat& m, std::int64_t p) {
        if (m.rows == 0 || m.cols == 0) {
            return 0;
        }

        std::vector<std::int64_t> a(static_cast<std::size_t>(m.rows) * m.cols);
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < m.cols; ++c) {
                a[static_cast<std::size_t>(r) * m.cols + c] = mod_norm(m.at(r, c), p);
            }
        }

        int rank = 0;
        for (int col = 0; col < m.cols && rank < m.rows; ++col) {
            int pivot = -1;
            for (int r = rank; r < m.rows; ++r) {
                if (a[static_cast<std::size_t>(r) * m.cols + col] != 0) {
                    pivot = r;
                    break;
                }
            }
            if (pivot < 0) {
                continue;
            }

            if (pivot != rank) {
                for (int c = 0; c < m.cols; ++c) {
                    std::swap(a[static_cast<std::size_t>(pivot) * m.cols + c],
                        a[static_cast<std::size_t>(rank) * m.cols + c]);
                }
            }

            const std::int64_t inv = mod_pow(a[static_cast<std::size_t>(rank) * m.cols + col], p - 2, p);
            for (int c = col; c < m.cols; ++c) {
                a[static_cast<std::size_t>(rank) * m.cols + c] =
                    (a[static_cast<std::size_t>(rank) * m.cols + c] * inv) % p;
            }

            for (int r = rank + 1; r < m.rows; ++r) {
                const std::int64_t f = a[static_cast<std::size_t>(r) * m.cols + col];
                if (f == 0) {
                    continue;
                }
                for (int c = col; c < m.cols; ++c) {
                    const std::int64_t idx = static_cast<std::int64_t>(r) * m.cols + c;
                    const std::int64_t ridx = static_cast<std::int64_t>(rank) * m.cols + c;
                    a[static_cast<std::size_t>(idx)] = mod_norm(a[static_cast<std::size_t>(idx)] - f * a[static_cast<std::size_t>(ridx)], p);
                }
            }
            ++rank;
        }

        return rank;
    }

    struct RrefData {
        int rank = 0;
        std::vector<std::uint32_t> flat;
    };

    RrefData rref_mod(const Mat& m, std::int64_t p) {
        RrefData out;
        if (m.rows == 0 || m.cols == 0) {
            return out;
        }

        std::vector<std::int64_t> a(static_cast<std::size_t>(m.rows) * m.cols);
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < m.cols; ++c) {
                a[static_cast<std::size_t>(r) * m.cols + c] = mod_norm(m.at(r, c), p);
            }
        }

        int rank = 0;
        for (int col = 0; col < m.cols && rank < m.rows; ++col) {
            int pivot = -1;
            for (int r = rank; r < m.rows; ++r) {
                if (a[static_cast<std::size_t>(r) * m.cols + col] != 0) {
                    pivot = r;
                    break;
                }
            }
            if (pivot < 0) {
                continue;
            }

            if (pivot != rank) {
                for (int c = 0; c < m.cols; ++c) {
                    std::swap(a[static_cast<std::size_t>(pivot) * m.cols + c],
                        a[static_cast<std::size_t>(rank) * m.cols + c]);
                }
            }

            const std::int64_t inv = mod_pow(a[static_cast<std::size_t>(rank) * m.cols + col], p - 2, p);
            for (int c = 0; c < m.cols; ++c) {
                a[static_cast<std::size_t>(rank) * m.cols + c] =
                    (a[static_cast<std::size_t>(rank) * m.cols + c] * inv) % p;
            }

            for (int r = 0; r < m.rows; ++r) {
                if (r == rank) {
                    continue;
                }
                const std::int64_t f = a[static_cast<std::size_t>(r) * m.cols + col];
                if (f == 0) {
                    continue;
                }
                for (int c = 0; c < m.cols; ++c) {
                    const std::size_t idx = static_cast<std::size_t>(r) * m.cols + c;
                    const std::size_t ridx = static_cast<std::size_t>(rank) * m.cols + c;
                    a[idx] = mod_norm(a[idx] - f * a[ridx], p);
                }
            }
            ++rank;
        }

        out.rank = rank;
        out.flat.reserve(static_cast<std::size_t>(rank) * m.cols);
        for (int r = 0; r < rank; ++r) {
            for (int c = 0; c < m.cols; ++c) {
                out.flat.push_back(static_cast<std::uint32_t>(a[static_cast<std::size_t>(r) * m.cols + c]));
            }
        }
        return out;
    }

    int rank_mat(const Mat& m) {
        // A nonzero rational minor can vanish modulo one prime.  Taking the
        // larger of two independent modular ranks makes all rank-based filters
        // substantially more robust while retaining fast integer arithmetic.
        return std::max(rank_mod(m, PRIME1), rank_mod(m, PRIME2));
    }

    std::string rref_key(const Mat& m) {
        const RrefData r1 = rref_mod(m, PRIME1);
        const RrefData r2 = rref_mod(m, PRIME2);

        std::string s;
        append_u32(s, static_cast<std::uint32_t>(m.cols));
        append_u32(s, static_cast<std::uint32_t>(r1.rank));
        append_u32(s, static_cast<std::uint32_t>(r2.rank));

        for (std::uint32_t v : r1.flat) {
            append_u32(s, v);
        }
        for (std::uint32_t v : r2.flat) {
            append_u32(s, v);
        }
        return s;
    }

    std::string pair_key_from_exact_keys(const std::string& e, const std::string& a) {
        std::string s;
        append_u32(s, static_cast<std::uint32_t>(e.size()));
        s += e;
        append_u32(s, static_cast<std::uint32_t>(a.size()));
        s += a;
        return s;
    }

    Mat prefix_rows(const Mat& m, int rows) {
        Mat out(rows, m.cols);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < m.cols; ++c) {
                out.at(r, c) = m.at(r, c);
            }
        }
        return out;
    }

    Mat append_matrix(const Mat& a, const Mat& b) {
        Mat out(a.rows + b.rows, a.cols);
        for (int r = 0; r < a.rows; ++r) {
            for (int c = 0; c < a.cols; ++c) {
                out.at(r, c) = a.at(r, c);
            }
        }
        for (int r = 0; r < b.rows; ++r) {
            for (int c = 0; c < a.cols; ++c) {
                out.at(a.rows + r, c) = b.at(r, c);
            }
        }
        return out;
    }

    Mat append_one_row(const Mat& a, const std::vector<int>& row) {
        Mat out(a.rows + 1, a.cols);
        for (int r = 0; r < a.rows; ++r) {
            for (int c = 0; c < a.cols; ++c) {
                out.at(r, c) = a.at(r, c);
            }
        }
        for (int c = 0; c < a.cols; ++c) {
            out.at(a.rows, c) = row[static_cast<std::size_t>(c)];
        }
        return out;
    }

    Mat first_cols(const Mat& m, int cols) {
        Mat out(m.rows, cols);
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                out.at(r, c) = m.at(r, c);
            }
        }
        return out;
    }

    std::vector<int> edge_dihedral_perm(int t, int polygon_n) {
        const int shift = t % polygon_n;
        const bool flip = t >= polygon_n;

        std::vector<int> p(static_cast<std::size_t>(polygon_n));
        for (int c = 0; c < polygon_n; ++c) {
            int old = (c - shift + polygon_n) % polygon_n;
            if (flip) {
                old = polygon_n - 1 - old;
            }
            p[static_cast<std::size_t>(c)] = old;
        }
        return p;
    }

    std::vector<int> angle_dihedral_perm(int t, int polygon_n) {
        const int shift = t % polygon_n;
        const bool flip = t >= polygon_n;

        std::vector<int> p(static_cast<std::size_t>(polygon_n));
        for (int c = 0; c < polygon_n; ++c) {
            int old = (c - shift + polygon_n) % polygon_n;
            if (flip && old != 0) {
                old = polygon_n - old;
            }
            p[static_cast<std::size_t>(c)] = old;
        }
        return p;
    }

    Mat permute_edge(const Mat& m, const std::vector<int>& p) {
        const int polygon_n = static_cast<int>(p.size());
        Mat out(m.rows, polygon_n);
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < polygon_n; ++c) {
                out.at(r, c) = m.at(r, p[static_cast<std::size_t>(c)]);
            }
        }
        return out;
    }

    Mat permute_angle(const Mat& m, const std::vector<int>& p) {
        const int polygon_n = static_cast<int>(p.size());
        Mat out(m.rows, polygon_n + 1);
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < polygon_n; ++c) {
                out.at(r, c) = m.at(r, p[static_cast<std::size_t>(c)]);
            }
            out.at(r, polygon_n) = m.at(r, polygon_n);
        }
        return out;
    }

    std::string canonical_edge_dihedral_key(const Mat& e, int polygon_n) {
        std::string best;
        bool has_best = false;
        for (int t = 0; t < 2 * polygon_n; ++t) {
            const Mat pe = permute_edge(e, edge_dihedral_perm(t, polygon_n));
            const std::string key = rref_key(pe);
            if (!has_best || key < best) {
                best = key;
                has_best = true;
            }
        }
        return best;
    }

    std::string canonical_angle_symmetric_key(const Mat& a, int polygon_n) {
        std::string best;
        bool has_best = false;

        // Enumerate distinct arrangements of the first n column vectors.  Using
        // the column vectors themselves, instead of n distinct indices, avoids
        // recomputing permutations that only swap identical columns.
        std::vector<std::vector<int>> columns(
            static_cast<std::size_t>(polygon_n),
            std::vector<int>(static_cast<std::size_t>(a.rows), 0));
        for (int c = 0; c < polygon_n; ++c) {
            for (int r = 0; r < a.rows; ++r) {
                columns[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] = a.at(r, c);
            }
        }
        std::sort(columns.begin(), columns.end());

        do {
            Mat pa(a.rows, polygon_n + 1);
            for (int c = 0; c < polygon_n; ++c) {
                for (int r = 0; r < a.rows; ++r) {
                    pa.at(r, c) = columns[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)];
                }
            }
            for (int r = 0; r < a.rows; ++r) {
                pa.at(r, polygon_n) = a.at(r, polygon_n);
            }
            const std::string key = rref_key(pa);
            if (!has_best || key < best) {
                best = key;
                has_best = true;
            }
        } while (std::next_permutation(columns.begin(), columns.end()));
        return best;
    }

    std::string canonical_angle_dihedral_key(const Mat& a, int polygon_n) {
        std::string best;
        bool has_best = false;
        for (int t = 0; t < 2 * polygon_n; ++t) {
            // In angle-only pu=1 mode the first n columns are reflected as a
            // whole.  The final constant column is always fixed.
            const Mat pa = permute_angle(a, edge_dihedral_perm(t, polygon_n));
            const std::string key = rref_key(pa);
            if (!has_best || key < best) {
                best = key;
                has_best = true;
            }
        }
        return best;
    }

    std::string canonical_angle_key(const Mat& a, AngleKeyMode mode, int polygon_n) {
        if (mode == AngleKeyMode::Symmetric) {
            return canonical_angle_symmetric_key(a, polygon_n);
        }
        if (mode == AngleKeyMode::Dihedral) {
            return canonical_angle_dihedral_key(a, polygon_n);
        }
        return rref_key(a);
    }

    std::string canonical_pair_dihedral_key(const Mat& e, const Mat& a, int polygon_n) {
        std::string best;
        bool has_best = false;
        for (int t = 0; t < 2 * polygon_n; ++t) {
            const Mat pe = permute_edge(e, edge_dihedral_perm(t, polygon_n));
            const Mat pa = permute_angle(a, angle_dihedral_perm(t, polygon_n));
            const std::string key = pair_key_from_exact_keys(rref_key(pe), rref_key(pa));
            if (!has_best || key < best) {
                best = key;
                has_best = true;
            }
        }
        return best;
    }

    std::vector<std::vector<int>> make_edge_rows(int polygon_n) {
        std::vector<std::vector<int>> rows;
        rows.reserve(static_cast<std::size_t>(polygon_n) * (polygon_n - 1) / 2);
        for (int i = 0; i < polygon_n; ++i) {
            for (int j = i + 1; j < polygon_n; ++j) {
                std::vector<int> row(static_cast<std::size_t>(polygon_n), 0);
                row[static_cast<std::size_t>(i)] = 1;
                row[static_cast<std::size_t>(j)] = -1;
                rows.push_back(row);
            }
        }
        return rows;
    }

    void angle_rows_rec(int pos, int rest, int polygon_n, std::vector<int>& row, std::vector<std::vector<int>>& rows) {
        if (pos == polygon_n - 1) {
            row[static_cast<std::size_t>(polygon_n - 1)] = rest;
            row[static_cast<std::size_t>(polygon_n)] = 2;
            rows.push_back(row);
            return;
        }
        for (int v = 0; v <= rest; ++v) {
            row[static_cast<std::size_t>(pos)] = v;
            angle_rows_rec(pos + 1, rest - v, polygon_n, row, rows);
        }
    }

    std::vector<std::vector<int>> make_angle_rows(int polygon_n, int vertex_order) {
        std::vector<std::vector<int>> rows;
        std::vector<int> row(static_cast<std::size_t>(polygon_n + 1), 0);
        angle_rows_rec(0, vertex_order, polygon_n, row, rows);
        return rows;
    }

    void set_row(Mat& m, int r, const std::vector<int>& row) {
        for (int c = 0; c < m.cols; ++c) {
            m.at(r, c) = row[static_cast<std::size_t>(c)];
        }
    }

    struct DSU {
        std::vector<int> parent;

        explicit DSU(int n) : parent(static_cast<std::size_t>(n)) {
            for (int i = 0; i < n; ++i) {
                parent[static_cast<std::size_t>(i)] = i;
            }
        }

        int find(int x) {
            int& p = parent[static_cast<std::size_t>(x)];
            if (p != x) {
                p = find(p);
            }
            return p;
        }

        void unite(int a, int b) {
            a = find(a);
            b = find(b);
            if (a != b) {
                parent[static_cast<std::size_t>(b)] = a;
            }
        }

        bool same(int a, int b) {
            return find(a) == find(b);
        }
    };

    bool passes_edge_consecutive_rule(const Mat& e, int polygon_n) {
        const int base_rank = e.rows;
        DSU dsu(polygon_n);

        for (int i = 0; i < polygon_n; ++i) {
            for (int j = i + 1; j < polygon_n; ++j) {
                std::vector<int> diff(static_cast<std::size_t>(polygon_n), 0);
                diff[static_cast<std::size_t>(i)] = 1;
                diff[static_cast<std::size_t>(j)] = -1;
                const Mat test = append_one_row(e, diff);
                if (rank_mat(test) == base_rank) {
                    dsu.unite(i, j);
                }
            }
        }

        std::vector<int> size(static_cast<std::size_t>(polygon_n), 0);
        for (int i = 0; i < polygon_n; ++i) {
            ++size[static_cast<std::size_t>(dsu.find(i))];
        }

        for (int i = 0; i < polygon_n; ++i) {
            const int u = i;
            const int a = (i + 1) % polygon_n;
            const int b = (i + 2) % polygon_n;
            const int v = (i + 3) % polygon_n;

            const bool a_single = size[static_cast<std::size_t>(dsu.find(a))] == 1;
            const bool b_single = size[static_cast<std::size_t>(dsu.find(b))] == 1;
            if (a_single && b_single && !dsu.same(u, v)) {
                return false;
            }
        }
        return true;
    }

    bool passes_angle_no_zero_col(const Mat& a, int polygon_n) {
        for (int c = 0; c < polygon_n; ++c) {
            bool any = false;
            for (int r = 0; r < a.rows; ++r) {
                if (a.at(r, c) != 0) {
                    any = true;
                    break;
                }
            }
            if (!any) {
                return false;
            }
        }
        return true;
    }

    bool passes_balance(const Mat& a, int polygon_n) {
        Mat angle_coefficients = first_cols(a, polygon_n);
        const int r0 = rank_mat(angle_coefficients);

        std::vector<int> ones(static_cast<std::size_t>(polygon_n), 1);
        const Mat with_ones = append_one_row(angle_coefficients, ones);
        return rank_mat(with_ones) == r0;
    }

    // ------------------------ Strong balance (mode 2) ------------------------
    //
    // Let C be every legal angle row [x,2] in the row space of A.  The required
    // positive integer weights exist exactly when positive rational weights exist.
    // With q_i = 1 + z_i, the equal-coordinate condition becomes
    //
    //     B*z = -B*1,  z >= 0,
    //
    // where column i of B is C_i(0:n-2)-C_i(n-1).  The phase-I revised simplex
    // below stores only an (n-1)-square basis and scans the completion on demand.

    struct ModRowSpace {
        int cols = 0;
        std::int64_t prime = 0;
        std::vector<std::int64_t> rref;
        std::vector<int> pivots;

        ModRowSpace() = default;
        ModRowSpace(const Mat& matrix, std::int64_t p) {
            build(matrix, p);
        }

        void build(const Mat& matrix, std::int64_t p) {
            cols = matrix.cols;
            prime = p;
            rref.assign(static_cast<std::size_t>(matrix.rows) * cols, 0);
            for (int r = 0; r < matrix.rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    rref[static_cast<std::size_t>(r) * cols + c] =
                        mod_norm(matrix.at(r, c), prime);
                }
            }

            pivots.clear();
            int rank = 0;
            for (int col = 0; col < cols && rank < matrix.rows; ++col) {
                int pivot = -1;
                for (int r = rank; r < matrix.rows; ++r) {
                    if (rref[static_cast<std::size_t>(r) * cols + col] != 0) {
                        pivot = r;
                        break;
                    }
                }
                if (pivot < 0) {
                    continue;
                }
                if (pivot != rank) {
                    for (int c = 0; c < cols; ++c) {
                        std::swap(rref[static_cast<std::size_t>(pivot) * cols + c],
                            rref[static_cast<std::size_t>(rank) * cols + c]);
                    }
                }

                const std::int64_t inverse = mod_pow(
                    rref[static_cast<std::size_t>(rank) * cols + col],
                    prime - 2,
                    prime);
                for (int c = 0; c < cols; ++c) {
                    rref[static_cast<std::size_t>(rank) * cols + c] =
                        rref[static_cast<std::size_t>(rank) * cols + c] * inverse % prime;
                }
                for (int r = 0; r < matrix.rows; ++r) {
                    if (r == rank) {
                        continue;
                    }
                    const std::int64_t factor =
                        rref[static_cast<std::size_t>(r) * cols + col];
                    if (factor == 0) {
                        continue;
                    }
                    for (int c = 0; c < cols; ++c) {
                        const std::size_t dst = static_cast<std::size_t>(r) * cols + c;
                        const std::size_t src = static_cast<std::size_t>(rank) * cols + c;
                        rref[dst] = mod_norm(rref[dst] - factor * rref[src], prime);
                    }
                }
                pivots.push_back(col);
                ++rank;
            }
            rref.resize(static_cast<std::size_t>(rank) * cols);
        }

        bool contains_with_scratch(
            const std::vector<int>& row,
            std::vector<std::int64_t>& residual) const {

            if (static_cast<int>(row.size()) != cols) {
                return false;
            }
            residual.assign(static_cast<std::size_t>(cols), 0);
            for (int c = 0; c < cols; ++c) {
                residual[static_cast<std::size_t>(c)] =
                    mod_norm(row[static_cast<std::size_t>(c)], prime);
            }
            for (std::size_t r = 0; r < pivots.size(); ++r) {
                const int pivot = pivots[r];
                const std::int64_t factor = residual[static_cast<std::size_t>(pivot)];
                if (factor == 0) {
                    continue;
                }
                for (int c = pivot; c < cols; ++c) {
                    residual[static_cast<std::size_t>(c)] = mod_norm(
                        residual[static_cast<std::size_t>(c)] -
                        factor * rref[r * static_cast<std::size_t>(cols) + c],
                        prime);
                }
            }
            for (std::int64_t value : residual) {
                if (value != 0) {
                    return false;
                }
            }
            return true;
        }
    };

    bool solve_dense_system(
        std::vector<long double> matrix,
        std::vector<long double> rhs,
        int dimension,
        std::vector<long double>& solution) {

        long double scale = 1.0L;
        for (long double value : matrix) {
            scale = std::max(scale, std::abs(value));
        }
        const long double pivot_tolerance =
            256.0L * std::numeric_limits<long double>::epsilon() * scale *
            std::max(1, dimension);

        for (int col = 0; col < dimension; ++col) {
            int pivot = col;
            long double best = 0.0L;
            for (int row = col; row < dimension; ++row) {
                const long double candidate = std::abs(
                    matrix[static_cast<std::size_t>(row) * dimension + col]);
                if (candidate > best) {
                    best = candidate;
                    pivot = row;
                }
            }
            if (best <= pivot_tolerance) {
                return false;
            }
            if (pivot != col) {
                for (int c = col; c < dimension; ++c) {
                    std::swap(matrix[static_cast<std::size_t>(pivot) * dimension + c],
                        matrix[static_cast<std::size_t>(col) * dimension + c]);
                }
                std::swap(rhs[static_cast<std::size_t>(pivot)],
                    rhs[static_cast<std::size_t>(col)]);
            }

            const long double divisor =
                matrix[static_cast<std::size_t>(col) * dimension + col];
            for (int c = col; c < dimension; ++c) {
                matrix[static_cast<std::size_t>(col) * dimension + c] /= divisor;
            }
            rhs[static_cast<std::size_t>(col)] /= divisor;

            for (int row = 0; row < dimension; ++row) {
                if (row == col) {
                    continue;
                }
                const long double factor =
                    matrix[static_cast<std::size_t>(row) * dimension + col];
                if (factor == 0.0L) {
                    continue;
                }
                for (int c = col; c < dimension; ++c) {
                    matrix[static_cast<std::size_t>(row) * dimension + c] -=
                        factor * matrix[static_cast<std::size_t>(col) * dimension + c];
                }
                rhs[static_cast<std::size_t>(row)] -=
                    factor * rhs[static_cast<std::size_t>(col)];
            }
        }

        solution = std::move(rhs);
        return true;
    }

    class StrongBalancePhaseOne {
    public:
        StrongBalancePhaseOne(
            const std::vector<std::vector<int>>& legal_rows,
            const std::vector<std::uint32_t>& completion,
            int polygon_n,
            std::vector<std::int64_t> target)
            : rows_(legal_rows),
            completion_(completion),
            n_(polygon_n),
            d_(polygon_n - 1),
            target_(std::move(target)) {
        }

        bool feasible() {
            if (d_ <= 0) {
                return true;
            }

            bool target_zero = true;
            for (std::int64_t value : target_) {
                target_zero = target_zero && value == 0;
            }
            if (target_zero) {
                return true; // Coefficient 1 on every completion row already works.
            }

            const std::size_t original_count = completion_.size();
            std::vector<std::size_t> basis(static_cast<std::size_t>(d_), 0);
            for (int i = 0; i < d_; ++i) {
                basis[static_cast<std::size_t>(i)] =
                    original_count + static_cast<std::size_t>(i);
            }

            constexpr std::size_t MAX_ITERATIONS = 1000000;
            std::vector<long double> basic_values;
            std::vector<long double> dual;
            for (std::size_t iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
                std::vector<long double> basis_matrix;
                build_basis_matrix(basis, basis_matrix);

                std::vector<long double> target_ld(static_cast<std::size_t>(d_), 0.0L);
                std::vector<long double> costs(static_cast<std::size_t>(d_), 0.0L);
                for (int i = 0; i < d_; ++i) {
                    target_ld[static_cast<std::size_t>(i)] =
                        static_cast<long double>(target_[static_cast<std::size_t>(i)]);
                    costs[static_cast<std::size_t>(i)] =
                        basis[static_cast<std::size_t>(i)] >= original_count ? 1.0L : 0.0L;
                }

                if (!solve_dense_system(basis_matrix, target_ld, d_, basic_values)) {
                    throw std::runtime_error("strong-balance phase-I basis became singular");
                }

                std::vector<long double> transposed(
                    static_cast<std::size_t>(d_) * d_, 0.0L);
                for (int r = 0; r < d_; ++r) {
                    for (int c = 0; c < d_; ++c) {
                        transposed[static_cast<std::size_t>(r) * d_ + c] =
                            basis_matrix[static_cast<std::size_t>(c) * d_ + r];
                    }
                }
                // solve_dense_system takes its matrix by value, so basis_matrix is
                // still the original basis and can be reused for the direction.
                if (!solve_dense_system(transposed, costs, d_, dual)) {
                    throw std::runtime_error("strong-balance dual basis became singular");
                }

                const long double dual_scale = vector_scale(dual);
                const long double reduced_tolerance =
                    4096.0L * std::numeric_limits<long double>::epsilon() *
                    std::max(1.0L, dual_scale * std::max(1, n_));
                const std::size_t no_entering =
                    original_count + static_cast<std::size_t>(d_);
                std::size_t entering = no_entering;
                for (std::size_t variable = 0; variable < no_entering; ++variable) {
                    if (std::find(basis.begin(), basis.end(), variable) != basis.end()) {
                        continue;
                    }
                    const long double cost = variable >= original_count ? 1.0L : 0.0L;
                    const long double reduced = cost - dot_column(dual, variable);
                    if (reduced < -reduced_tolerance) {
                        entering = variable; // Bland's smallest-index rule.
                        break;
                    }
                }

                if (entering == no_entering) {
                    return certify_optimum(basis, basic_values, dual);
                }

                std::vector<long double> entering_column;
                column(entering, entering_column);
                std::vector<long double> direction;
                if (!solve_dense_system(
                    basis_matrix, entering_column, d_, direction)) {
                    throw std::runtime_error("strong-balance entering direction failed");
                }

                const long double value_scale = vector_scale(basic_values);
                const long double direction_tolerance =
                    4096.0L * std::numeric_limits<long double>::epsilon() *
                    std::max(1.0L, vector_scale(direction));
                int leaving_row = -1;
                long double best_ratio = std::numeric_limits<long double>::infinity();
                for (int row = 0; row < d_; ++row) {
                    const long double direction_value =
                        direction[static_cast<std::size_t>(row)];
                    if (direction_value <= direction_tolerance) {
                        continue;
                    }

                    long double value = basic_values[static_cast<std::size_t>(row)];
                    const long double feasibility_tolerance =
                        8192.0L * std::numeric_limits<long double>::epsilon() *
                        std::max(1.0L, value_scale);
                    if (value < -feasibility_tolerance) {
                        throw std::runtime_error(
                            "strong-balance phase-I lost primal feasibility");
                    }
                    if (value < 0.0L) {
                        value = 0.0L;
                    }

                    const long double ratio = value / direction_value;
                    const long double tie_tolerance =
                        8192.0L * std::numeric_limits<long double>::epsilon() *
                        std::max({ 1.0L, std::abs(ratio), std::abs(best_ratio) });
                    if (leaving_row < 0 || ratio < best_ratio - tie_tolerance ||
                        (std::abs(ratio - best_ratio) <= tie_tolerance &&
                            basis[static_cast<std::size_t>(row)] <
                            basis[static_cast<std::size_t>(leaving_row)])) {
                        leaving_row = row;
                        best_ratio = ratio;
                    }
                }
                if (leaving_row < 0) {
                    throw std::runtime_error(
                        "strong-balance phase-I reported an unbounded descent");
                }
                basis[static_cast<std::size_t>(leaving_row)] = entering;
            }

            throw std::runtime_error(
                "strong-balance phase-I iteration budget was exhausted");
        }

    private:
        const std::vector<std::vector<int>>& rows_;
        const std::vector<std::uint32_t>& completion_;
        int n_ = 0;
        int d_ = 0;
        std::vector<std::int64_t> target_;

        static long double vector_scale(const std::vector<long double>& values) {
            long double scale = 0.0L;
            for (long double value : values) {
                scale = std::max(scale, std::abs(value));
            }
            return scale;
        }

        void column(std::size_t variable, std::vector<long double>& output) const {
            output.assign(static_cast<std::size_t>(d_), 0.0L);
            if (variable < completion_.size()) {
                const std::vector<int>& row = rows_[completion_[variable]];
                const int reference = row[static_cast<std::size_t>(n_ - 1)];
                for (int c = 0; c < d_; ++c) {
                    output[static_cast<std::size_t>(c)] = static_cast<long double>(
                        row[static_cast<std::size_t>(c)] - reference);
                }
                return;
            }

            const int artificial =
                static_cast<int>(variable - completion_.size());
            const std::int64_t target_value =
                target_[static_cast<std::size_t>(artificial)];
            output[static_cast<std::size_t>(artificial)] =
                target_value < 0 ? -1.0L : 1.0L;
        }

        long double dot_column(
            const std::vector<long double>& dual,
            std::size_t variable) const {

            if (variable < completion_.size()) {
                const std::vector<int>& row = rows_[completion_[variable]];
                const int reference = row[static_cast<std::size_t>(n_ - 1)];
                long double dot = 0.0L;
                for (int c = 0; c < d_; ++c) {
                    dot += dual[static_cast<std::size_t>(c)] *
                        static_cast<long double>(
                            row[static_cast<std::size_t>(c)] - reference);
                }
                return dot;
            }

            const int artificial =
                static_cast<int>(variable - completion_.size());
            return dual[static_cast<std::size_t>(artificial)] *
                (target_[static_cast<std::size_t>(artificial)] < 0 ? -1.0L : 1.0L);
        }

        void build_basis_matrix(
            const std::vector<std::size_t>& basis,
            std::vector<long double>& matrix) const {

            matrix.assign(static_cast<std::size_t>(d_) * d_, 0.0L);
            std::vector<long double> values;
            for (int col = 0; col < d_; ++col) {
                column(basis[static_cast<std::size_t>(col)], values);
                for (int row = 0; row < d_; ++row) {
                    matrix[static_cast<std::size_t>(row) * d_ + col] =
                        values[static_cast<std::size_t>(row)];
                }
            }
        }

        bool certify_optimum(
            const std::vector<std::size_t>& basis,
            const std::vector<long double>& basic_values,
            std::vector<long double> dual) const {

            const std::size_t original_count = completion_.size();
            long double phase_one_value = 0.0L;
            long double target_scale = 1.0L;
            for (std::int64_t value : target_) {
                target_scale = std::max(
                    target_scale, std::abs(static_cast<long double>(value)));
            }
            for (int i = 0; i < d_; ++i) {
                if (basis[static_cast<std::size_t>(i)] >= original_count) {
                    phase_one_value += std::max(
                        0.0L, basic_values[static_cast<std::size_t>(i)]);
                }
            }

            const long double certificate_tolerance =
                65536.0L * std::numeric_limits<long double>::epsilon() *
                target_scale * std::max(1, d_);
            if (phase_one_value <= certificate_tolerance) {
                std::vector<long double> residual(static_cast<std::size_t>(d_), 0.0L);
                for (int c = 0; c < d_; ++c) {
                    residual[static_cast<std::size_t>(c)] =
                        static_cast<long double>(target_[static_cast<std::size_t>(c)]);
                }
                std::vector<long double> values;
                for (int i = 0; i < d_; ++i) {
                    if (basis[static_cast<std::size_t>(i)] >= original_count) {
                        continue;
                    }
                    column(basis[static_cast<std::size_t>(i)], values);
                    const long double coefficient =
                        std::max(0.0L, basic_values[static_cast<std::size_t>(i)]);
                    for (int c = 0; c < d_; ++c) {
                        residual[static_cast<std::size_t>(c)] -=
                            coefficient * values[static_cast<std::size_t>(c)];
                    }
                }
                if (vector_scale(residual) <= certificate_tolerance) {
                    return true;
                }
                throw std::runtime_error(
                    "strong-balance feasible certificate was numerically ambiguous");
            }

            const long double dual_norm = vector_scale(dual);
            if (dual_norm == 0.0L) {
                throw std::runtime_error(
                    "strong-balance infeasible certificate had zero dual norm");
            }
            for (long double& value : dual) {
                value /= dual_norm;
            }

            long double max_column_dot =
                -std::numeric_limits<long double>::infinity();
            for (std::size_t variable = 0; variable < original_count; ++variable) {
                max_column_dot = std::max(
                    max_column_dot, dot_column(dual, variable));
            }
            long double target_dot = 0.0L;
            for (int c = 0; c < d_; ++c) {
                target_dot += dual[static_cast<std::size_t>(c)] *
                    static_cast<long double>(target_[static_cast<std::size_t>(c)]);
            }

            const long double dual_tolerance =
                131072.0L * std::numeric_limits<long double>::epsilon() *
                std::max(1.0L, target_scale) * std::max(1, d_);
            if (max_column_dot <= dual_tolerance && target_dot > dual_tolerance) {
                return false;
            }
            throw std::runtime_error(
                "strong-balance infeasible certificate was numerically ambiguous");
        }
    };

    bool passes_strong_balance(
        const Mat& angle,
        const std::vector<std::vector<int>>& legal_rows,
        int polygon_n) {

        ModRowSpace first(angle, PRIME1);
        ModRowSpace second(angle, PRIME2);
        std::vector<std::int64_t> first_residual;
        std::vector<std::int64_t> second_residual;
        first_residual.reserve(static_cast<std::size_t>(polygon_n + 1));
        second_residual.reserve(static_cast<std::size_t>(polygon_n + 1));

        std::vector<std::uint32_t> completion;
        completion.reserve(std::min<std::size_t>(legal_rows.size(), 4096));
        std::vector<std::int64_t> sums(
            static_cast<std::size_t>(polygon_n - 1), 0);

        for (std::size_t index = 0; index < legal_rows.size(); ++index) {
            const std::vector<int>& row = legal_rows[index];
            if (!first.contains_with_scratch(row, first_residual) ||
                !second.contains_with_scratch(row, second_residual)) {
                continue;
            }
            if (index > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "strong-balance completion row index exceeds uint32 capacity");
            }
            completion.push_back(static_cast<std::uint32_t>(index));

            const int reference = row[static_cast<std::size_t>(polygon_n - 1)];
            for (int c = 0; c < polygon_n - 1; ++c) {
                const std::int64_t delta = static_cast<std::int64_t>(
                    row[static_cast<std::size_t>(c)]) - reference;
                std::int64_t& sum = sums[static_cast<std::size_t>(c)];
                if ((delta > 0 &&
                    sum > std::numeric_limits<std::int64_t>::max() - delta) ||
                    (delta < 0 &&
                        sum < std::numeric_limits<std::int64_t>::min() - delta)) {
                    throw std::runtime_error(
                        "strong-balance completion sum exceeds int64 capacity");
                }
                sum += delta;
            }
        }

        if (completion.empty()) {
            throw std::runtime_error(
                "strong-balance completion unexpectedly contained no legal rows");
        }

        std::vector<std::int64_t> target(
            static_cast<std::size_t>(polygon_n - 1), 0);
        for (int c = 0; c < polygon_n - 1; ++c) {
            const std::int64_t sum = sums[static_cast<std::size_t>(c)];
            if (sum == std::numeric_limits<std::int64_t>::min()) {
                throw std::runtime_error(
                    "strong-balance completion sum exceeds int64 capacity");
            }
            target[static_cast<std::size_t>(c)] = -sum;
        }

        StrongBalancePhaseOne solver(
            legal_rows, completion, polygon_n, std::move(target));
        return solver.feasible();
    }

    void enumerate_edge_rec(
        const std::vector<std::vector<int>>& rows,
        int start,
        int left,
        int depth,
        Mat& cur,
        bool enforce_consecutive_rule,
        bool use_dihedral_key,
        int polygon_n,
        std::unordered_set<std::string>& local_seen,
        std::vector<Candidate>& local_out) {

        if (left == 0) {
            if (enforce_consecutive_rule && !passes_edge_consecutive_rule(cur, polygon_n)) {
                return;
            }
            const std::string key = use_dihedral_key ? canonical_edge_dihedral_key(cur, polygon_n) : rref_key(cur);
            if (local_seen.insert(key).second) {
                local_out.push_back(Candidate{ cur, key });
            }
            return;
        }

        const int n = static_cast<int>(rows.size());
        for (int i = start; i <= n - left; ++i) {
            set_row(cur, depth, rows[static_cast<std::size_t>(i)]);
            const Mat pref = prefix_rows(cur, depth + 1);
            if (rank_mat(pref) == depth + 1) {
                enumerate_edge_rec(rows, i + 1, left - 1, depth + 1, cur,
                    enforce_consecutive_rule, use_dihedral_key, polygon_n, local_seen, local_out);
            }
        }
    }

    std::vector<Mat> merge_candidates(const std::vector<std::vector<Candidate>>& locals) {
        std::unordered_set<std::string> seen;
        std::vector<Mat> out;
        for (const auto& bucket : locals) {
            for (const Candidate& c : bucket) {
                if (seen.insert(c.key).second) {
                    out.push_back(c.mat);
                }
            }
        }
        return out;
    }

    void enumerate_set_partitions_rec(
        int pos,
        int max_label,
        int polygon_n,
        int target_blocks,
        std::vector<int>& labels,
        std::vector<std::vector<int>>& partitions) {

        const int used_blocks = max_label + 1;
        const int remaining = polygon_n - pos;
        if (used_blocks > target_blocks || used_blocks + remaining < target_blocks) {
            return;
        }
        if (pos == polygon_n) {
            if (used_blocks == target_blocks) {
                partitions.push_back(labels);
            }
            return;
        }

        // Restricted-growth strings enumerate every unlabeled set partition
        // exactly once.
        const int max_next = std::min(max_label + 1, target_blocks - 1);
        for (int label = 0; label <= max_next; ++label) {
            labels[static_cast<std::size_t>(pos)] = label;
            enumerate_set_partitions_rec(
                pos + 1,
                std::max(max_label, label),
                polygon_n,
                target_blocks,
                labels,
                partitions);
        }
    }

    Mat partition_to_edge_basis(const std::vector<int>& labels, int target_rank, int target_blocks) {
        const int polygon_n = static_cast<int>(labels.size());
        Mat edge(target_rank, polygon_n);
        std::vector<int> anchor(static_cast<std::size_t>(target_blocks), -1);
        int row = 0;

        for (int i = 0; i < polygon_n; ++i) {
            const int block = labels[static_cast<std::size_t>(i)];
            int& first = anchor[static_cast<std::size_t>(block)];
            if (first < 0) {
                first = i;
            }
            else {
                edge.at(row, first) = 1;
                edge.at(row, i) = -1;
                ++row;
            }
        }
        return edge;
    }

    std::vector<Mat> generate_edges(
        int target_rank,
        bool enforce_consecutive_rule,
        bool use_dihedral_key,
        int polygon_n,
        int num_threads) {
        if (target_rank <= 0 || target_rank > polygon_n - 1) {
            return {};
        }

        // A span of rows e_i-e_j is determined exactly by the connected
        // components of the equality graph.  Rank r therefore corresponds to a
        // set partition with n-r blocks.  Enumerating those partitions avoids the
        // large number of different forests that generate the same row space.
        const int target_blocks = polygon_n - target_rank;
        std::vector<std::vector<int>> partitions;
        std::vector<int> labels(static_cast<std::size_t>(polygon_n), 0);
        labels[0] = 0;
        enumerate_set_partitions_rec(
            1, 0, polygon_n, target_blocks, labels, partitions);

        const long long work_items = static_cast<long long>(partitions.size());
        const int buckets = choose_workers(num_threads, work_items);
        std::vector<std::vector<Candidate>> locals(static_cast<std::size_t>(buckets));
        std::vector<std::unordered_set<std::string>> local_seen(static_cast<std::size_t>(buckets));

        parallel_for_dynamic(0, work_items, buckets, [&](int tid, long long idx) {
            Mat edge = partition_to_edge_basis(
                partitions[static_cast<std::size_t>(idx)], target_rank, target_blocks);
            if (enforce_consecutive_rule && !passes_edge_consecutive_rule(edge, polygon_n)) {
                return;
            }
            const std::string key = use_dihedral_key
                ? canonical_edge_dihedral_key(edge, polygon_n)
                : rref_key(edge);
            if (local_seen[static_cast<std::size_t>(tid)].insert(key).second) {
                locals[static_cast<std::size_t>(tid)].push_back(Candidate{ std::move(edge), key });
            }
            });

        return merge_candidates(locals);
    }

    int zero_angle_columns(const Mat& a, int polygon_n);

    void enumerate_angle_rec(
        const std::vector<std::vector<int>>& rows,
        int start,
        int left,
        int depth,
        Mat& cur,
        AngleKeyMode key_mode,
        bool pingheng,
        int polygon_n,
        int vertex_order,
        int target_rank,
        std::unordered_set<std::string>& local_seen,
        std::vector<Candidate>& local_out) {

        if (left == 0) {
            if (!passes_angle_no_zero_col(cur, polygon_n)) {
                return;
            }
            if (pingheng && !passes_balance(cur, polygon_n)) {
                return;
            }

            const std::string key = canonical_angle_key(cur, key_mode, polygon_n);
            if (local_seen.insert(key).second) {
                local_out.push_back(Candidate{ cur, key });
            }
            return;
        }

        const int n = static_cast<int>(rows.size());
        for (int i = start; i <= n - left; ++i) {
            set_row(cur, depth, rows[static_cast<std::size_t>(i)]);
            const Mat pref = prefix_rows(cur, depth + 1);
            if (rank_mat(pref) == depth + 1) {
                const int rows_left_after = left - 1;
                if (zero_angle_columns(pref, polygon_n) > vertex_order * rows_left_after) {
                    continue;
                }
                if (pingheng) {
                    Mat coefficients = first_cols(pref, polygon_n);
                    const std::vector<int> ones(static_cast<std::size_t>(polygon_n), 1);
                    if (rank_mat(append_one_row(coefficients, ones)) > target_rank) {
                        continue;
                    }
                }
                enumerate_angle_rec(rows, i + 1, left - 1, depth + 1, cur,
                    key_mode, pingheng, polygon_n, vertex_order, target_rank,
                    local_seen, local_out);
            }
        }
    }

    int zero_angle_columns(const Mat& a, int polygon_n) {
        int zeros = 0;
        for (int c = 0; c < polygon_n; ++c) {
            bool any = false;
            for (int r = 0; r < a.rows; ++r) {
                if (a.at(r, c) != 0) {
                    any = true;
                    break;
                }
            }
            if (!any) {
                ++zeros;
            }
        }
        return zeros;
    }

    std::vector<Mat> generate_angles_orbit_bfs(
        int target_rank,
        AngleKeyMode key_mode,
        bool pingheng,
        int polygon_n,
        int vertex_order,
        int num_threads) {

        if (target_rank <= 0 || target_rank > polygon_n) {
            return {};
        }

        const auto rows = make_angle_rows(polygon_n, vertex_order);
        const int nRows = static_cast<int>(rows.size());

        std::vector<Mat> states;
        states.emplace_back(0, polygon_n + 1);

        for (int depth = 0; depth < target_rank; ++depth) {
            if (states.empty()) {
                break;
            }

            const int next_dim = depth + 1;
            const int rows_left_after = target_rank - next_dim;
            const long long total = static_cast<long long>(states.size()) * nRows;
            const int buckets = choose_workers(num_threads, total);

            std::vector<std::vector<Candidate>> locals(static_cast<std::size_t>(buckets));
            std::vector<std::unordered_set<std::string>> local_seen(static_cast<std::size_t>(buckets));
            std::vector<std::unordered_set<std::string>> local_exact_seen(static_cast<std::size_t>(buckets));
            const std::vector<int> ones(static_cast<std::size_t>(polygon_n), 1);

            parallel_for_dynamic(0, total, buckets, [&](int tid, long long idx) {
                const std::size_t state_idx = static_cast<std::size_t>(idx / nRows);
                const std::size_t row_idx = static_cast<std::size_t>(idx % nRows);

                const Mat next = append_one_row(states[state_idx], rows[row_idx]);
                if (rank_mat(next) != next_dim) {
                    return;
                }

                if (zero_angle_columns(next, polygon_n) > vertex_order * rows_left_after) {
                    return;
                }

                if (pingheng) {
                    Mat angle_coefficients = first_cols(next, polygon_n);
                    const Mat with_ones = append_one_row(angle_coefficients, ones);
                    if (rank_mat(with_ones) > target_rank) {
                        return;
                    }
                }

                if (next_dim == target_rank) {
                    if (!passes_angle_no_zero_col(next, polygon_n)) {
                        return;
                    }
                    if (pingheng && !passes_balance(next, polygon_n)) {
                        return;
                    }
                }

                const std::string direct_key = rref_key(next);
                if (!local_exact_seen[static_cast<std::size_t>(tid)].insert(direct_key).second) {
                    return;
                }

                const std::string key = canonical_angle_key(next, key_mode, polygon_n);
                if (local_seen[static_cast<std::size_t>(tid)].insert(key).second) {
                    locals[static_cast<std::size_t>(tid)].push_back(Candidate{ next, key });
                }
                });

            states = merge_candidates(locals);
            mexPrintf(">>> C++: angle-only BFS rank %d representatives = %llu\n",
                next_dim, static_cast<unsigned long long>(states.size()));
            mexEvalString("drawnow;");
        }

        return states;
    }

    std::vector<Mat> generate_full_rank_angle_representative(int polygon_n, int vertex_order) {
        Mat a(polygon_n, polygon_n + 1);
        for (int r = 0; r < polygon_n; ++r) {
            a.at(r, r) = vertex_order;
            a.at(r, polygon_n) = 2;
        }
        return std::vector<Mat>{a};
    }

    std::vector<Mat> generate_angles(
        int target_rank,
        AngleKeyMode key_mode,
        bool pingheng,
        int polygon_n,
        int vertex_order,
        int num_threads) {

        if (target_rank <= 0 || target_rank > polygon_n) {
            return {};
        }

        if (target_rank == polygon_n) {
            return generate_full_rank_angle_representative(polygon_n, vertex_order);
        }

        if (key_mode != AngleKeyMode::Exact) {
            return generate_angles_orbit_bfs(
                target_rank, key_mode, pingheng, polygon_n, vertex_order, num_threads);
        }

        const auto rows = make_angle_rows(polygon_n, vertex_order);
        const int n = static_cast<int>(rows.size());
        const int k = target_rank;
        if (k > n) {
            return {};
        }

        const long long work_items = static_cast<long long>(n - k + 1);
        const int buckets = choose_workers(num_threads, work_items);
        std::vector<std::vector<Candidate>> locals(static_cast<std::size_t>(buckets));
        std::vector<std::unordered_set<std::string>> local_seen(static_cast<std::size_t>(buckets));

        parallel_for_dynamic(0, work_items, buckets, [&](int tid, long long first_ll) {
            const int first = static_cast<int>(first_ll);
            Mat cur(k, polygon_n + 1);
            set_row(cur, 0, rows[static_cast<std::size_t>(first)]);
            enumerate_angle_rec(rows, first + 1, k - 1, 1, cur,
                key_mode, pingheng, polygon_n, vertex_order, target_rank,
                local_seen[static_cast<std::size_t>(tid)],
                locals[static_cast<std::size_t>(tid)]);
            });

        return merge_candidates(locals);
    }

    struct StrongBalanceFilterResult {
        std::vector<Mat> angles;
        std::uint64_t tested = 0;
        std::uint64_t rejected = 0;
    };

    StrongBalanceFilterResult retain_strongly_balanced(
        std::vector<Mat> angles,
        int target_rank,
        int polygon_n,
        int vertex_order,
        int num_threads) {

        StrongBalanceFilterResult result;
        result.tested = static_cast<std::uint64_t>(angles.size());
        if (angles.empty()) {
            result.angles = std::move(angles);
            return result;
        }

        // Rank n is the complete legal angle hyperplane.  Its completion is
        // invariant under every column permutation, so weight 1 on every row is
        // already a strong-balance certificate.  Avoid materializing that possibly
        // very large completion in this case.
        if (target_rank == polygon_n) {
            result.angles = std::move(angles);
            return result;
        }

        const std::vector<std::vector<int>> legal_rows =
            make_angle_rows(polygon_n, vertex_order);
        const int workers = choose_workers(
            num_threads, static_cast<long long>(angles.size()));
        std::vector<unsigned char> keep(angles.size(), 0);
        std::vector<std::exception_ptr> failures(static_cast<std::size_t>(workers));
        std::atomic<bool> abort(false);

        parallel_for_dynamic(
            0,
            static_cast<long long>(angles.size()),
            workers,
            [&](int tid, long long index_ll) {
                if (abort.load(std::memory_order_relaxed)) {
                    return;
                }
                const std::size_t index = static_cast<std::size_t>(index_ll);
                try {
                    keep[index] = passes_strong_balance(
                        angles[index], legal_rows, polygon_n) ? 1 : 0;
                }
                catch (...) {
                    failures[static_cast<std::size_t>(tid)] =
                        std::current_exception();
                    abort.store(true, std::memory_order_relaxed);
                }
            });

        for (const std::exception_ptr& failure : failures) {
            if (failure) {
                std::rethrow_exception(failure);
            }
        }

        result.angles.reserve(angles.size());
        for (std::size_t index = 0; index < angles.size(); ++index) {
            if (keep[index] != 0) {
                result.angles.push_back(std::move(angles[index]));
            }
        }
        result.rejected = result.tested -
            static_cast<std::uint64_t>(result.angles.size());
        return result;
    }

    bool contains_rowspace(const Mat& candidate, const Mat& known) {
        if (known.rows == 0) {
            return true;
        }
        const Mat stacked = append_matrix(candidate, known);
        return rank_mat(stacked) == candidate.rows;
    }

    std::vector<PairCandidate> merge_pairs(const std::vector<std::vector<PairCandidate>>& locals) {
        std::unordered_set<std::string> seen;
        std::vector<PairCandidate> out;
        for (const auto& bucket : locals) {
            for (const PairCandidate& p : bucket) {
                if (seen.insert(p.key).second) {
                    out.push_back(p);
                }
            }
        }
        return out;
    }

    std::vector<std::vector<std::string>> make_edge_orbit_keys(
        const std::vector<Mat>& edges,
        int polygon_n,
        int num_threads) {

        std::vector<std::vector<std::string>> keys(
            edges.size(), std::vector<std::string>(static_cast<std::size_t>(2 * polygon_n)));
        const int workers = choose_workers(num_threads, static_cast<long long>(edges.size()));
        parallel_for_dynamic(0, static_cast<long long>(edges.size()), workers, [&](int, long long i_ll) {
            const std::size_t i = static_cast<std::size_t>(i_ll);
            for (int t = 0; t < 2 * polygon_n; ++t) {
                keys[i][static_cast<std::size_t>(t)] = rref_key(
                    permute_edge(edges[i], edge_dihedral_perm(t, polygon_n)));
            }
            });
        return keys;
    }

    std::vector<std::vector<std::string>> make_joint_angle_orbit_keys(
        const std::vector<Mat>& angles,
        int polygon_n,
        int num_threads) {

        std::vector<std::vector<std::string>> keys(
            angles.size(), std::vector<std::string>(static_cast<std::size_t>(2 * polygon_n)));
        const int workers = choose_workers(num_threads, static_cast<long long>(angles.size()));
        parallel_for_dynamic(0, static_cast<long long>(angles.size()), workers, [&](int, long long i_ll) {
            const std::size_t i = static_cast<std::size_t>(i_ll);
            for (int t = 0; t < 2 * polygon_n; ++t) {
                keys[i][static_cast<std::size_t>(t)] = rref_key(
                    permute_angle(angles[i], angle_dihedral_perm(t, polygon_n)));
            }
            });
        return keys;
    }

    std::string canonical_pair_from_orbit_keys(
        const std::vector<std::string>& edge_keys,
        const std::vector<std::string>& angle_keys) {

        std::string best;
        bool has_best = false;
        for (std::size_t t = 0; t < edge_keys.size(); ++t) {
            const std::string key = pair_key_from_exact_keys(edge_keys[t], angle_keys[t]);
            if (!has_best || key < best) {
                best = key;
                has_best = true;
            }
        }
        return best;
    }

    std::vector<PairCandidate> generate_pairs(
        const std::vector<Mat>& edges,
        const std::vector<Mat>& angles,
        const std::vector<KnownPair>& known,
        int polygon_n,
        int num_threads) {

        const std::size_t nE = edges.size();
        const std::size_t nA = angles.size();
        const std::size_t nK = known.size();

        if (nE == 0 || nA == 0) {
            return {};
        }

        // RREFs of transformed matrices depend only on the individual edge or
        // angle candidate.  Precomputing them avoids 4*n RREF computations for
        // every element of the Cartesian product.
        const auto edge_orbit_keys = make_edge_orbit_keys(edges, polygon_n, num_threads);
        const auto angle_orbit_keys = make_joint_angle_orbit_keys(angles, polygon_n, num_threads);

        std::vector<std::vector<unsigned char>> e_has(nE, std::vector<unsigned char>(nK, 0));
        std::vector<std::vector<unsigned char>> a_has(nA, std::vector<unsigned char>(nK, 0));

        if (nK > 0) {
            const int e_workers = choose_workers(num_threads, static_cast<long long>(nE));
            parallel_for_dynamic(0, static_cast<long long>(nE), e_workers, [&](int, long long i_ll) {
                const std::size_t i = static_cast<std::size_t>(i_ll);
                for (std::size_t k = 0; k < nK; ++k) {
                    e_has[i][k] = contains_rowspace(edges[i], known[k].edge) ? 1 : 0;
                }
                });

            const int a_workers = choose_workers(num_threads, static_cast<long long>(nA));
            parallel_for_dynamic(0, static_cast<long long>(nA), a_workers, [&](int, long long i_ll) {
                const std::size_t i = static_cast<std::size_t>(i_ll);
                for (std::size_t k = 0; k < nK; ++k) {
                    a_has[i][k] = contains_rowspace(angles[i], known[k].angle) ? 1 : 0;
                }
                });
        }

        const long long total = static_cast<long long>(nE) * static_cast<long long>(nA);
        const int buckets = choose_workers(num_threads, total);
        std::vector<std::vector<PairCandidate>> locals(static_cast<std::size_t>(buckets));
        std::vector<std::unordered_set<std::string>> local_seen(static_cast<std::size_t>(buckets));

        parallel_for_dynamic(0, total, buckets, [&](int tid, long long idx) {
            const std::size_t ei = static_cast<std::size_t>(idx / static_cast<long long>(nA));
            const std::size_t ai = static_cast<std::size_t>(idx % static_cast<long long>(nA));

            bool degenerate = false;
            for (std::size_t k = 0; k < nK; ++k) {
                if (e_has[ei][k] && a_has[ai][k]) {
                    degenerate = true;
                    break;
                }
            }
            if (degenerate) {
                return;
            }

            const std::string key = canonical_pair_from_orbit_keys(
                edge_orbit_keys[ei], angle_orbit_keys[ai]);
            if (local_seen[static_cast<std::size_t>(tid)].insert(key).second) {
                locals[static_cast<std::size_t>(tid)].push_back(PairCandidate{ edges[ei], angles[ai], key });
            }
            });

        return merge_pairs(locals);
    }

    int scalar_int(const mxArray* a, const char* name) {
        if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) {
            mexErrMsgIdAndTxt("all_basis_generator:BadInput", "%s must be a real scalar double.", name);
        }
        const double value = mxGetScalar(a);
        if (!std::isfinite(value) || value != std::floor(value) ||
            value < static_cast<double>(std::numeric_limits<int>::min()) ||
            value > static_cast<double>(std::numeric_limits<int>::max())) {
            mexErrMsgIdAndTxt("all_basis_generator:BadInput", "%s must be a finite integer-valued scalar.", name);
        }
        return static_cast<int>(value);
    }

    int checked_int_value(double value, const char* name) {
        if (!std::isfinite(value) || value != std::floor(value) ||
            value < static_cast<double>(std::numeric_limits<int>::min()) ||
            value > static_cast<double>(std::numeric_limits<int>::max())) {
            mexErrMsgIdAndTxt(
                "all_basis_generator:BadKnownPool",
                "%s must contain finite integer-valued entries representable as C++ int.",
                name);
        }
        return static_cast<int>(value);
    }

    std::vector<KnownPair> parse_known(
        const mxArray* eArr,
        const mxArray* aArr,
        const mxArray* eRowsArr,
        const mxArray* aRowsArr,
        int polygon_n) {

        if (!mxIsDouble(eArr) || !mxIsDouble(aArr) || !mxIsDouble(eRowsArr) || !mxIsDouble(aRowsArr)) {
            mexErrMsgIdAndTxt("all_basis_generator:BadKnownPool", "Known_E, Known_A and row-count arrays must be double arrays.");
        }

        const mwSize* ed = mxGetDimensions(eArr);
        const mwSize* ad = mxGetDimensions(aArr);
        const mwSize e_nd = mxGetNumberOfDimensions(eArr);
        const mwSize a_nd = mxGetNumberOfDimensions(aArr);

        if (e_nd < 2 || a_nd < 2 || ed[1] != static_cast<mwSize>(polygon_n) ||
            ad[1] != static_cast<mwSize>(polygon_n + 1)) {
            mexErrMsgIdAndTxt("all_basis_generator:BadKnownPool",
                "Known_E must be maxRows x n x N and Known_A must be maxRows x (n+1) x N.");
        }

        const mwSize nE = (e_nd >= 3) ? ed[2] : 1;
        const mwSize nA = (a_nd >= 3) ? ad[2] : 1;
        const mwSize nR = std::min(mxGetNumberOfElements(eRowsArr), mxGetNumberOfElements(aRowsArr));
        const mwSize n = std::min(std::min(nE, nA), nR);

        const double* ep = mxGetPr(eArr);
        const double* ap = mxGetPr(aArr);
        const double* erp = mxGetPr(eRowsArr);
        const double* arp = mxGetPr(aRowsArr);

        std::vector<KnownPair> out;
        out.reserve(static_cast<std::size_t>(n));

        for (mwSize k = 0; k < n; ++k) {
            const int rE = checked_int_value(erp[k], "Known_E_rows");
            const int rA = checked_int_value(arp[k], "Known_A_rows");
            if (rE <= 0 || rA <= 0) {
                continue;
            }
            if (static_cast<mwSize>(rE) > ed[0] || static_cast<mwSize>(rA) > ad[0]) {
                mexErrMsgIdAndTxt("all_basis_generator:BadKnownPool", "Known row count exceeds the allocated known_pool array size.");
            }

            Mat e(rE, polygon_n);
            Mat a(rA, polygon_n + 1);

            for (int r = 0; r < rE; ++r) {
                for (int c = 0; c < polygon_n; ++c) {
                    const mwSize idx = static_cast<mwSize>(r) + ed[0] *
                        (static_cast<mwSize>(c) + static_cast<mwSize>(polygon_n) * k);
                    e.at(r, c) = checked_int_value(ep[idx], "Known_E");
                }
            }
            for (int r = 0; r < rA; ++r) {
                for (int c = 0; c < polygon_n + 1; ++c) {
                    const mwSize idx = static_cast<mwSize>(r) + ad[0] *
                        (static_cast<mwSize>(c) + static_cast<mwSize>(polygon_n + 1) * k);
                    a.at(r, c) = checked_int_value(ap[idx], "Known_A");
                }
            }

            out.push_back(KnownPair{ e, a });
        }

        return out;
    }

    mxArray* mats_to_mx(const std::vector<Mat>& mats, int rows, int cols) {
        if (mats.empty()) {
            return mxCreateDoubleMatrix(0, static_cast<mwSize>(cols), mxREAL);
        }

        const mwSize dims[3] = {
            static_cast<mwSize>(rows),
            static_cast<mwSize>(cols),
            static_cast<mwSize>(mats.size())
        };
        mxArray* out = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        double* p = mxGetPr(out);

        for (std::size_t k = 0; k < mats.size(); ++k) {
            const Mat& m = mats[k];
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    const mwSize idx = static_cast<mwSize>(r) +
                        static_cast<mwSize>(rows) * (static_cast<mwSize>(c) + static_cast<mwSize>(cols) * static_cast<mwSize>(k));
                    p[idx] = static_cast<double>(m.at(r, c));
                }
            }
        }
        return out;
    }

    mxArray* pair_edges_to_mx(const std::vector<PairCandidate>& pairs, int rows, int polygon_n) {
        if (pairs.empty()) {
            return mxCreateDoubleMatrix(0, static_cast<mwSize>(polygon_n), mxREAL);
        }

        const mwSize dims[3] = {
            static_cast<mwSize>(rows),
            static_cast<mwSize>(polygon_n),
            static_cast<mwSize>(pairs.size())
        };
        mxArray* out = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        double* p = mxGetPr(out);
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < polygon_n; ++c) {
                    const mwSize idx = static_cast<mwSize>(r) +
                        static_cast<mwSize>(rows) * (static_cast<mwSize>(c) +
                            static_cast<mwSize>(polygon_n) * static_cast<mwSize>(k));
                    p[idx] = static_cast<double>(pairs[k].edge.at(r, c));
                }
            }
        }
        return out;
    }

    mxArray* pair_angles_to_mx(const std::vector<PairCandidate>& pairs, int rows, int polygon_n) {
        if (pairs.empty()) {
            return mxCreateDoubleMatrix(0, static_cast<mwSize>(polygon_n + 1), mxREAL);
        }

        const mwSize dims[3] = {
            static_cast<mwSize>(rows),
            static_cast<mwSize>(polygon_n + 1),
            static_cast<mwSize>(pairs.size())
        };
        mxArray* out = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        double* p = mxGetPr(out);
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < polygon_n + 1; ++c) {
                    const mwSize idx = static_cast<mwSize>(r) +
                        static_cast<mwSize>(rows) * (static_cast<mwSize>(c) +
                            static_cast<mwSize>(polygon_n + 1) * static_cast<mwSize>(k));
                    p[idx] = static_cast<double>(pairs[k].angle.at(r, c));
                }
            }
        }
        return out;
    }

} // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 10 && nrhs != 11) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput",
            "Usage: [E_results,A_results] = all_basis_generator(Target_RE,Target_RA,pingheng,Num_Threads,n,jieshu[,pu],Known_E,Known_A,Known_E_rows,Known_A_rows)");
    }
    if (nlhs < 1 || nlhs > 2) {
        mexErrMsgIdAndTxt("all_basis_generator:BadOutput", "One or two outputs are required.");
    }

    const int targetRE = scalar_int(prhs[0], "Target_RE");
    const int targetRA = scalar_int(prhs[1], "Target_RA");
    const int balanceMode = scalar_int(prhs[2], "pingheng");
    const bool weakBalance = balanceMode >= 1;
    const int numThreads = std::max(1, scalar_int(prhs[3], "Num_Threads"));
    const int polygonN = scalar_int(prhs[4], "n");
    const int vertexOrder = scalar_int(prhs[5], "jieshu");
    const bool hasPu = nrhs == 11;
    const int pu = hasPu ? scalar_int(prhs[6], "pu") : 0;
    const int knownOffset = hasPu ? 7 : 6;

    if (targetRE < 0 || targetRA < 0) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "Target ranks must be non-negative.");
    }
    if (balanceMode < 0 || balanceMode > 2) {
        mexErrMsgIdAndTxt(
            "all_basis_generator:BadInput",
            "pingheng must be 0 (none), 1 (weak), or 2 (weak plus strong).");
    }
    if (polygonN < 3) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "n must be at least 3.");
    }
    if (vertexOrder < 1) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "jieshu must be a positive integer.");
    }
    if (pu != 0 && pu != 1) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "pu must be 0 or 1.");
    }
    if (targetRE > polygonN - 1) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "Target_RE cannot exceed n-1.");
    }
    if (targetRA > polygonN) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "Target_RA cannot exceed n.");
    }
    if (targetRE == 0 && targetRA == 0) {
        plhs[0] = mats_to_mx({}, 0, polygonN);
        if (nlhs > 1) {
            plhs[1] = mats_to_mx({}, 0, polygonN + 1);
        }
        return;
    }

    const std::vector<KnownPair> baseKnown = parse_known(
        prhs[knownOffset],
        prhs[knownOffset + 1],
        prhs[knownOffset + 2],
        prhs[knownOffset + 3],
        polygonN);

    mexPrintf(
        ">>> C++: generating candidates for n=%d, jieshu=%d, pu=%d, pingheng=%d...\n",
        polygonN, vertexOrder, pu, balanceMode);
    mexEvalString("drawnow;");

    if (targetRE == 0) {
        const AngleKeyMode angleOnlyMode = pu == 0
            ? AngleKeyMode::Symmetric
            : AngleKeyMode::Dihedral;
        std::vector<Mat> angles = generate_angles(
            targetRA, angleOnlyMode, weakBalance,
            polygonN, vertexOrder, numThreads);
        if (balanceMode == 2 && targetRA > 0) {
            try {
                StrongBalanceFilterResult filtered = retain_strongly_balanced(
                    std::move(angles), targetRA,
                    polygonN, vertexOrder, numThreads);
                mexPrintf(
                    ">>> C++: strong-balance tested = %llu, rejected = %llu\n",
                    static_cast<unsigned long long>(filtered.tested),
                    static_cast<unsigned long long>(filtered.rejected));
                angles = std::move(filtered.angles);
            }
            catch (const std::exception& error) {
                mexErrMsgIdAndTxt(
                    "all_basis_generator:StrongBalanceFailure",
                    "Strong-balance filtering failed: %s",
                    error.what());
            }
            catch (...) {
                mexErrMsgIdAndTxt(
                    "all_basis_generator:StrongBalanceFailure",
                    "Strong-balance filtering failed with an unknown C++ exception.");
            }
        }
        mexPrintf(">>> C++: angle-only result count = %llu\n", static_cast<unsigned long long>(angles.size()));
        plhs[0] = mats_to_mx(std::vector<Mat>{}, 0, polygonN);
        if (nlhs > 1) {
            plhs[1] = mats_to_mx(angles, targetRA, polygonN + 1);
        }
        return;
    }

    if (targetRA == 0) {
        const std::vector<Mat> edges = generate_edges(
            targetRE, vertexOrder == 3, true, polygonN, numThreads);
        mexPrintf(">>> C++: edge-only result count = %llu\n", static_cast<unsigned long long>(edges.size()));
        plhs[0] = mats_to_mx(edges, targetRE, polygonN);
        if (nlhs > 1) {
            plhs[1] = mats_to_mx(std::vector<Mat>{}, 0, polygonN + 1);
        }
        return;
    }

    const std::vector<Mat> edges = generate_edges(
        targetRE, vertexOrder == 3, true, polygonN, numThreads);
    mexPrintf(">>> C++: edge D%d representatives before joint pairing = %llu\n",
        polygonN, static_cast<unsigned long long>(edges.size()));
    mexEvalString("drawnow;");

    std::vector<Mat> angles = generate_angles(
        targetRA, AngleKeyMode::Exact, weakBalance,
        polygonN, vertexOrder, numThreads);
    mexPrintf(">>> C++: angle representatives = %llu\n", static_cast<unsigned long long>(angles.size()));
    mexEvalString("drawnow;");

    if (balanceMode == 2) {
        try {
            StrongBalanceFilterResult filtered = retain_strongly_balanced(
                std::move(angles), targetRA,
                polygonN, vertexOrder, numThreads);
            mexPrintf(
                ">>> C++: strong-balance tested = %llu, rejected = %llu, retained = %llu\n",
                static_cast<unsigned long long>(filtered.tested),
                static_cast<unsigned long long>(filtered.rejected),
                static_cast<unsigned long long>(filtered.angles.size()));
            angles = std::move(filtered.angles);
        }
        catch (const std::exception& error) {
            mexErrMsgIdAndTxt(
                "all_basis_generator:StrongBalanceFailure",
                "Strong-balance filtering failed: %s",
                error.what());
        }
        catch (...) {
            mexErrMsgIdAndTxt(
                "all_basis_generator:StrongBalanceFailure",
                "Strong-balance filtering failed with an unknown C++ exception.");
        }
        mexEvalString("drawnow;");
    }

    const std::vector<KnownPair>& known = baseKnown;
    mexPrintf(">>> C++: known_pool degeneracy entries = %llu\n", static_cast<unsigned long long>(known.size()));
    mexEvalString("drawnow;");

    const std::vector<PairCandidate> pairs = generate_pairs(edges, angles, known, polygonN, numThreads);
    mexPrintf(">>> C++: joint result count = %llu\n", static_cast<unsigned long long>(pairs.size()));
    mexEvalString("drawnow;");

    plhs[0] = pair_edges_to_mx(pairs, targetRE, polygonN);
    if (nlhs > 1) {
        plhs[1] = pair_angles_to_mx(pairs, targetRA, polygonN);
    }
}






//no Ç¿Æ½ºâ

/*#include "mex.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <atomic>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

    constexpr std::int64_t PRIME1 = 1000000007LL;
    constexpr std::int64_t PRIME2 = 1000000009LL;

    struct Mat {
        int rows = 0;
        int cols = 0;
        std::vector<int> x;

        Mat() = default;
        Mat(int r, int c) : rows(r), cols(c), x(static_cast<std::size_t>(r)* c, 0) {}

        int& at(int r, int c) {
            return x[static_cast<std::size_t>(r) * cols + c];
        }

        int at(int r, int c) const {
            return x[static_cast<std::size_t>(r) * cols + c];
        }
    };

    struct Candidate {
        Mat mat;
        std::string key;
    };

    struct PairCandidate {
        Mat edge;
        Mat angle;
        std::string key;
    };

    struct KnownPair {
        Mat edge;
        Mat angle;
    };

    enum class AngleKeyMode {
        Exact,
        Symmetric,
        Dihedral
    };

    int choose_workers(int requested, long long work_items) {
        if (work_items <= 1) {
            return 1;
        }
        const long long capped = std::min<long long>(std::max(1, requested), work_items);
        return static_cast<int>(capped);
    }

    template <class Fn>
    void parallel_for_dynamic(long long begin, long long end, int workers, Fn fn) {
        if (workers <= 1 || end <= begin + 1) {
            for (long long i = begin; i < end; ++i) {
                fn(0, i);
            }
            return;
        }

        std::atomic<long long> next(begin);
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(workers));

        for (int tid = 0; tid < workers; ++tid) {
            threads.emplace_back([&, tid]() {
                while (true) {
                    const long long i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= end) {
                        break;
                    }
                    fn(tid, i);
                }
                });
        }

        for (std::thread& t : threads) {
            t.join();
        }
    }

    std::int64_t mod_norm(std::int64_t v, std::int64_t p) {
        v %= p;
        return v < 0 ? v + p : v;
    }

    std::int64_t mod_pow(std::int64_t a, std::int64_t e, std::int64_t p) {
        std::int64_t r = 1;
        a = mod_norm(a, p);
        while (e > 0) {
            if (e & 1) {
                r = (r * a) % p;
            }
            a = (a * a) % p;
            e >>= 1;
        }
        return r;
    }

    void append_u32(std::string& s, std::uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            s.push_back(static_cast<char>((v >> (8 * i)) & 0xffu));
        }
    }

    int rank_mod(const Mat& m, std::int64_t p) {
        if (m.rows == 0 || m.cols == 0) {
            return 0;
        }

        std::vector<std::int64_t> a(static_cast<std::size_t>(m.rows) * m.cols);
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < m.cols; ++c) {
                a[static_cast<std::size_t>(r) * m.cols + c] = mod_norm(m.at(r, c), p);
            }
        }

        int rank = 0;
        for (int col = 0; col < m.cols && rank < m.rows; ++col) {
            int pivot = -1;
            for (int r = rank; r < m.rows; ++r) {
                if (a[static_cast<std::size_t>(r) * m.cols + col] != 0) {
                    pivot = r;
                    break;
                }
            }
            if (pivot < 0) {
                continue;
            }

            if (pivot != rank) {
                for (int c = 0; c < m.cols; ++c) {
                    std::swap(a[static_cast<std::size_t>(pivot) * m.cols + c],
                        a[static_cast<std::size_t>(rank) * m.cols + c]);
                }
            }

            const std::int64_t inv = mod_pow(a[static_cast<std::size_t>(rank) * m.cols + col], p - 2, p);
            for (int c = col; c < m.cols; ++c) {
                a[static_cast<std::size_t>(rank) * m.cols + c] =
                    (a[static_cast<std::size_t>(rank) * m.cols + c] * inv) % p;
            }

            for (int r = rank + 1; r < m.rows; ++r) {
                const std::int64_t f = a[static_cast<std::size_t>(r) * m.cols + col];
                if (f == 0) {
                    continue;
                }
                for (int c = col; c < m.cols; ++c) {
                    const std::int64_t idx = static_cast<std::int64_t>(r) * m.cols + c;
                    const std::int64_t ridx = static_cast<std::int64_t>(rank) * m.cols + c;
                    a[static_cast<std::size_t>(idx)] = mod_norm(a[static_cast<std::size_t>(idx)] - f * a[static_cast<std::size_t>(ridx)], p);
                }
            }
            ++rank;
        }

        return rank;
    }

    struct RrefData {
        int rank = 0;
        std::vector<std::uint32_t> flat;
    };

    RrefData rref_mod(const Mat& m, std::int64_t p) {
        RrefData out;
        if (m.rows == 0 || m.cols == 0) {
            return out;
        }

        std::vector<std::int64_t> a(static_cast<std::size_t>(m.rows) * m.cols);
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < m.cols; ++c) {
                a[static_cast<std::size_t>(r) * m.cols + c] = mod_norm(m.at(r, c), p);
            }
        }

        int rank = 0;
        for (int col = 0; col < m.cols && rank < m.rows; ++col) {
            int pivot = -1;
            for (int r = rank; r < m.rows; ++r) {
                if (a[static_cast<std::size_t>(r) * m.cols + col] != 0) {
                    pivot = r;
                    break;
                }
            }
            if (pivot < 0) {
                continue;
            }

            if (pivot != rank) {
                for (int c = 0; c < m.cols; ++c) {
                    std::swap(a[static_cast<std::size_t>(pivot) * m.cols + c],
                        a[static_cast<std::size_t>(rank) * m.cols + c]);
                }
            }

            const std::int64_t inv = mod_pow(a[static_cast<std::size_t>(rank) * m.cols + col], p - 2, p);
            for (int c = 0; c < m.cols; ++c) {
                a[static_cast<std::size_t>(rank) * m.cols + c] =
                    (a[static_cast<std::size_t>(rank) * m.cols + c] * inv) % p;
            }

            for (int r = 0; r < m.rows; ++r) {
                if (r == rank) {
                    continue;
                }
                const std::int64_t f = a[static_cast<std::size_t>(r) * m.cols + col];
                if (f == 0) {
                    continue;
                }
                for (int c = 0; c < m.cols; ++c) {
                    const std::size_t idx = static_cast<std::size_t>(r) * m.cols + c;
                    const std::size_t ridx = static_cast<std::size_t>(rank) * m.cols + c;
                    a[idx] = mod_norm(a[idx] - f * a[ridx], p);
                }
            }
            ++rank;
        }

        out.rank = rank;
        out.flat.reserve(static_cast<std::size_t>(rank) * m.cols);
        for (int r = 0; r < rank; ++r) {
            for (int c = 0; c < m.cols; ++c) {
                out.flat.push_back(static_cast<std::uint32_t>(a[static_cast<std::size_t>(r) * m.cols + c]));
            }
        }
        return out;
    }

    int rank_mat(const Mat& m) {
        // A nonzero rational minor can vanish modulo one prime.  Taking the
        // larger of two independent modular ranks makes all rank-based filters
        // substantially more robust while retaining fast integer arithmetic.
        return std::max(rank_mod(m, PRIME1), rank_mod(m, PRIME2));
    }

    std::string rref_key(const Mat& m) {
        const RrefData r1 = rref_mod(m, PRIME1);
        const RrefData r2 = rref_mod(m, PRIME2);

        std::string s;
        append_u32(s, static_cast<std::uint32_t>(m.cols));
        append_u32(s, static_cast<std::uint32_t>(r1.rank));
        append_u32(s, static_cast<std::uint32_t>(r2.rank));

        for (std::uint32_t v : r1.flat) {
            append_u32(s, v);
        }
        for (std::uint32_t v : r2.flat) {
            append_u32(s, v);
        }
        return s;
    }

    std::string pair_key_from_exact_keys(const std::string& e, const std::string& a) {
        std::string s;
        append_u32(s, static_cast<std::uint32_t>(e.size()));
        s += e;
        append_u32(s, static_cast<std::uint32_t>(a.size()));
        s += a;
        return s;
    }

    Mat prefix_rows(const Mat& m, int rows) {
        Mat out(rows, m.cols);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < m.cols; ++c) {
                out.at(r, c) = m.at(r, c);
            }
        }
        return out;
    }

    Mat append_matrix(const Mat& a, const Mat& b) {
        Mat out(a.rows + b.rows, a.cols);
        for (int r = 0; r < a.rows; ++r) {
            for (int c = 0; c < a.cols; ++c) {
                out.at(r, c) = a.at(r, c);
            }
        }
        for (int r = 0; r < b.rows; ++r) {
            for (int c = 0; c < a.cols; ++c) {
                out.at(a.rows + r, c) = b.at(r, c);
            }
        }
        return out;
    }

    Mat append_one_row(const Mat& a, const std::vector<int>& row) {
        Mat out(a.rows + 1, a.cols);
        for (int r = 0; r < a.rows; ++r) {
            for (int c = 0; c < a.cols; ++c) {
                out.at(r, c) = a.at(r, c);
            }
        }
        for (int c = 0; c < a.cols; ++c) {
            out.at(a.rows, c) = row[static_cast<std::size_t>(c)];
        }
        return out;
    }

    Mat first_cols(const Mat& m, int cols) {
        Mat out(m.rows, cols);
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                out.at(r, c) = m.at(r, c);
            }
        }
        return out;
    }

    std::vector<int> edge_dihedral_perm(int t, int polygon_n) {
        const int shift = t % polygon_n;
        const bool flip = t >= polygon_n;

        std::vector<int> p(static_cast<std::size_t>(polygon_n));
        for (int c = 0; c < polygon_n; ++c) {
            int old = (c - shift + polygon_n) % polygon_n;
            if (flip) {
                old = polygon_n - 1 - old;
            }
            p[static_cast<std::size_t>(c)] = old;
        }
        return p;
    }

    std::vector<int> angle_dihedral_perm(int t, int polygon_n) {
        const int shift = t % polygon_n;
        const bool flip = t >= polygon_n;

        std::vector<int> p(static_cast<std::size_t>(polygon_n));
        for (int c = 0; c < polygon_n; ++c) {
            int old = (c - shift + polygon_n) % polygon_n;
            if (flip && old != 0) {
                old = polygon_n - old;
            }
            p[static_cast<std::size_t>(c)] = old;
        }
        return p;
    }

    Mat permute_edge(const Mat& m, const std::vector<int>& p) {
        const int polygon_n = static_cast<int>(p.size());
        Mat out(m.rows, polygon_n);
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < polygon_n; ++c) {
                out.at(r, c) = m.at(r, p[static_cast<std::size_t>(c)]);
            }
        }
        return out;
    }

    Mat permute_angle(const Mat& m, const std::vector<int>& p) {
        const int polygon_n = static_cast<int>(p.size());
        Mat out(m.rows, polygon_n + 1);
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < polygon_n; ++c) {
                out.at(r, c) = m.at(r, p[static_cast<std::size_t>(c)]);
            }
            out.at(r, polygon_n) = m.at(r, polygon_n);
        }
        return out;
    }

    std::string canonical_edge_dihedral_key(const Mat& e, int polygon_n) {
        std::string best;
        bool has_best = false;
        for (int t = 0; t < 2 * polygon_n; ++t) {
            const Mat pe = permute_edge(e, edge_dihedral_perm(t, polygon_n));
            const std::string key = rref_key(pe);
            if (!has_best || key < best) {
                best = key;
                has_best = true;
            }
        }
        return best;
    }

    std::string canonical_angle_symmetric_key(const Mat& a, int polygon_n) {
        std::string best;
        bool has_best = false;

        // Enumerate distinct arrangements of the first n column vectors.  Using
        // the column vectors themselves, instead of n distinct indices, avoids
        // recomputing permutations that only swap identical columns.
        std::vector<std::vector<int>> columns(
            static_cast<std::size_t>(polygon_n),
            std::vector<int>(static_cast<std::size_t>(a.rows), 0));
        for (int c = 0; c < polygon_n; ++c) {
            for (int r = 0; r < a.rows; ++r) {
                columns[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] = a.at(r, c);
            }
        }
        std::sort(columns.begin(), columns.end());

        do {
            Mat pa(a.rows, polygon_n + 1);
            for (int c = 0; c < polygon_n; ++c) {
                for (int r = 0; r < a.rows; ++r) {
                    pa.at(r, c) = columns[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)];
                }
            }
            for (int r = 0; r < a.rows; ++r) {
                pa.at(r, polygon_n) = a.at(r, polygon_n);
            }
            const std::string key = rref_key(pa);
            if (!has_best || key < best) {
                best = key;
                has_best = true;
            }
        } while (std::next_permutation(columns.begin(), columns.end()));
        return best;
    }

    std::string canonical_angle_dihedral_key(const Mat& a, int polygon_n) {
        std::string best;
        bool has_best = false;
        for (int t = 0; t < 2 * polygon_n; ++t) {
            // In angle-only pu=1 mode the first n columns are reflected as a
            // whole.  The final constant column is always fixed.
            const Mat pa = permute_angle(a, edge_dihedral_perm(t, polygon_n));
            const std::string key = rref_key(pa);
            if (!has_best || key < best) {
                best = key;
                has_best = true;
            }
        }
        return best;
    }

    std::string canonical_angle_key(const Mat& a, AngleKeyMode mode, int polygon_n) {
        if (mode == AngleKeyMode::Symmetric) {
            return canonical_angle_symmetric_key(a, polygon_n);
        }
        if (mode == AngleKeyMode::Dihedral) {
            return canonical_angle_dihedral_key(a, polygon_n);
        }
        return rref_key(a);
    }

    std::string canonical_pair_dihedral_key(const Mat& e, const Mat& a, int polygon_n) {
        std::string best;
        bool has_best = false;
        for (int t = 0; t < 2 * polygon_n; ++t) {
            const Mat pe = permute_edge(e, edge_dihedral_perm(t, polygon_n));
            const Mat pa = permute_angle(a, angle_dihedral_perm(t, polygon_n));
            const std::string key = pair_key_from_exact_keys(rref_key(pe), rref_key(pa));
            if (!has_best || key < best) {
                best = key;
                has_best = true;
            }
        }
        return best;
    }

    std::vector<std::vector<int>> make_edge_rows(int polygon_n) {
        std::vector<std::vector<int>> rows;
        rows.reserve(static_cast<std::size_t>(polygon_n) * (polygon_n - 1) / 2);
        for (int i = 0; i < polygon_n; ++i) {
            for (int j = i + 1; j < polygon_n; ++j) {
                std::vector<int> row(static_cast<std::size_t>(polygon_n), 0);
                row[static_cast<std::size_t>(i)] = 1;
                row[static_cast<std::size_t>(j)] = -1;
                rows.push_back(row);
            }
        }
        return rows;
    }

    void angle_rows_rec(int pos, int rest, int polygon_n, std::vector<int>& row, std::vector<std::vector<int>>& rows) {
        if (pos == polygon_n - 1) {
            row[static_cast<std::size_t>(polygon_n - 1)] = rest;
            row[static_cast<std::size_t>(polygon_n)] = 2;
            rows.push_back(row);
            return;
        }
        for (int v = 0; v <= rest; ++v) {
            row[static_cast<std::size_t>(pos)] = v;
            angle_rows_rec(pos + 1, rest - v, polygon_n, row, rows);
        }
    }

    std::vector<std::vector<int>> make_angle_rows(int polygon_n, int vertex_order) {
        std::vector<std::vector<int>> rows;
        std::vector<int> row(static_cast<std::size_t>(polygon_n + 1), 0);
        angle_rows_rec(0, vertex_order, polygon_n, row, rows);
        return rows;
    }

    void set_row(Mat& m, int r, const std::vector<int>& row) {
        for (int c = 0; c < m.cols; ++c) {
            m.at(r, c) = row[static_cast<std::size_t>(c)];
        }
    }

    struct DSU {
        std::vector<int> parent;

        explicit DSU(int n) : parent(static_cast<std::size_t>(n)) {
            for (int i = 0; i < n; ++i) {
                parent[static_cast<std::size_t>(i)] = i;
            }
        }

        int find(int x) {
            int& p = parent[static_cast<std::size_t>(x)];
            if (p != x) {
                p = find(p);
            }
            return p;
        }

        void unite(int a, int b) {
            a = find(a);
            b = find(b);
            if (a != b) {
                parent[static_cast<std::size_t>(b)] = a;
            }
        }

        bool same(int a, int b) {
            return find(a) == find(b);
        }
    };

    bool passes_edge_consecutive_rule(const Mat& e, int polygon_n) {
        const int base_rank = e.rows;
        DSU dsu(polygon_n);

        for (int i = 0; i < polygon_n; ++i) {
            for (int j = i + 1; j < polygon_n; ++j) {
                std::vector<int> diff(static_cast<std::size_t>(polygon_n), 0);
                diff[static_cast<std::size_t>(i)] = 1;
                diff[static_cast<std::size_t>(j)] = -1;
                const Mat test = append_one_row(e, diff);
                if (rank_mat(test) == base_rank) {
                    dsu.unite(i, j);
                }
            }
        }

        std::vector<int> size(static_cast<std::size_t>(polygon_n), 0);
        for (int i = 0; i < polygon_n; ++i) {
            ++size[static_cast<std::size_t>(dsu.find(i))];
        }

        for (int i = 0; i < polygon_n; ++i) {
            const int u = i;
            const int a = (i + 1) % polygon_n;
            const int b = (i + 2) % polygon_n;
            const int v = (i + 3) % polygon_n;

            const bool a_single = size[static_cast<std::size_t>(dsu.find(a))] == 1;
            const bool b_single = size[static_cast<std::size_t>(dsu.find(b))] == 1;
            if (a_single && b_single && !dsu.same(u, v)) {
                return false;
            }
        }
        return true;
    }

    bool passes_angle_no_zero_col(const Mat& a, int polygon_n) {
        for (int c = 0; c < polygon_n; ++c) {
            bool any = false;
            for (int r = 0; r < a.rows; ++r) {
                if (a.at(r, c) != 0) {
                    any = true;
                    break;
                }
            }
            if (!any) {
                return false;
            }
        }
        return true;
    }

    bool passes_balance(const Mat& a, int polygon_n) {
        Mat angle_coefficients = first_cols(a, polygon_n);
        const int r0 = rank_mat(angle_coefficients);

        std::vector<int> ones(static_cast<std::size_t>(polygon_n), 1);
        const Mat with_ones = append_one_row(angle_coefficients, ones);
        return rank_mat(with_ones) == r0;
    }

    void enumerate_edge_rec(
        const std::vector<std::vector<int>>& rows,
        int start,
        int left,
        int depth,
        Mat& cur,
        bool enforce_consecutive_rule,
        bool use_dihedral_key,
        int polygon_n,
        std::unordered_set<std::string>& local_seen,
        std::vector<Candidate>& local_out) {

        if (left == 0) {
            if (enforce_consecutive_rule && !passes_edge_consecutive_rule(cur, polygon_n)) {
                return;
            }
            const std::string key = use_dihedral_key ? canonical_edge_dihedral_key(cur, polygon_n) : rref_key(cur);
            if (local_seen.insert(key).second) {
                local_out.push_back(Candidate{ cur, key });
            }
            return;
        }

        const int n = static_cast<int>(rows.size());
        for (int i = start; i <= n - left; ++i) {
            set_row(cur, depth, rows[static_cast<std::size_t>(i)]);
            const Mat pref = prefix_rows(cur, depth + 1);
            if (rank_mat(pref) == depth + 1) {
                enumerate_edge_rec(rows, i + 1, left - 1, depth + 1, cur,
                    enforce_consecutive_rule, use_dihedral_key, polygon_n, local_seen, local_out);
            }
        }
    }

    std::vector<Mat> merge_candidates(const std::vector<std::vector<Candidate>>& locals) {
        std::unordered_set<std::string> seen;
        std::vector<Mat> out;
        for (const auto& bucket : locals) {
            for (const Candidate& c : bucket) {
                if (seen.insert(c.key).second) {
                    out.push_back(c.mat);
                }
            }
        }
        return out;
    }

    void enumerate_set_partitions_rec(
        int pos,
        int max_label,
        int polygon_n,
        int target_blocks,
        std::vector<int>& labels,
        std::vector<std::vector<int>>& partitions) {

        const int used_blocks = max_label + 1;
        const int remaining = polygon_n - pos;
        if (used_blocks > target_blocks || used_blocks + remaining < target_blocks) {
            return;
        }
        if (pos == polygon_n) {
            if (used_blocks == target_blocks) {
                partitions.push_back(labels);
            }
            return;
        }

        // Restricted-growth strings enumerate every unlabeled set partition
        // exactly once.
        const int max_next = std::min(max_label + 1, target_blocks - 1);
        for (int label = 0; label <= max_next; ++label) {
            labels[static_cast<std::size_t>(pos)] = label;
            enumerate_set_partitions_rec(
                pos + 1,
                std::max(max_label, label),
                polygon_n,
                target_blocks,
                labels,
                partitions);
        }
    }

    Mat partition_to_edge_basis(const std::vector<int>& labels, int target_rank, int target_blocks) {
        const int polygon_n = static_cast<int>(labels.size());
        Mat edge(target_rank, polygon_n);
        std::vector<int> anchor(static_cast<std::size_t>(target_blocks), -1);
        int row = 0;

        for (int i = 0; i < polygon_n; ++i) {
            const int block = labels[static_cast<std::size_t>(i)];
            int& first = anchor[static_cast<std::size_t>(block)];
            if (first < 0) {
                first = i;
            }
            else {
                edge.at(row, first) = 1;
                edge.at(row, i) = -1;
                ++row;
            }
        }
        return edge;
    }

    std::vector<Mat> generate_edges(
        int target_rank,
        bool enforce_consecutive_rule,
        bool use_dihedral_key,
        int polygon_n,
        int num_threads) {
        if (target_rank <= 0 || target_rank > polygon_n - 1) {
            return {};
        }

        // A span of rows e_i-e_j is determined exactly by the connected
        // components of the equality graph.  Rank r therefore corresponds to a
        // set partition with n-r blocks.  Enumerating those partitions avoids the
        // large number of different forests that generate the same row space.
        const int target_blocks = polygon_n - target_rank;
        std::vector<std::vector<int>> partitions;
        std::vector<int> labels(static_cast<std::size_t>(polygon_n), 0);
        labels[0] = 0;
        enumerate_set_partitions_rec(
            1, 0, polygon_n, target_blocks, labels, partitions);

        const long long work_items = static_cast<long long>(partitions.size());
        const int buckets = choose_workers(num_threads, work_items);
        std::vector<std::vector<Candidate>> locals(static_cast<std::size_t>(buckets));
        std::vector<std::unordered_set<std::string>> local_seen(static_cast<std::size_t>(buckets));

        parallel_for_dynamic(0, work_items, buckets, [&](int tid, long long idx) {
            Mat edge = partition_to_edge_basis(
                partitions[static_cast<std::size_t>(idx)], target_rank, target_blocks);
            if (enforce_consecutive_rule && !passes_edge_consecutive_rule(edge, polygon_n)) {
                return;
            }
            const std::string key = use_dihedral_key
                ? canonical_edge_dihedral_key(edge, polygon_n)
                : rref_key(edge);
            if (local_seen[static_cast<std::size_t>(tid)].insert(key).second) {
                locals[static_cast<std::size_t>(tid)].push_back(Candidate{ std::move(edge), key });
            }
            });

        return merge_candidates(locals);
    }

    int zero_angle_columns(const Mat& a, int polygon_n);

    void enumerate_angle_rec(
        const std::vector<std::vector<int>>& rows,
        int start,
        int left,
        int depth,
        Mat& cur,
        AngleKeyMode key_mode,
        bool pingheng,
        int polygon_n,
        int vertex_order,
        int target_rank,
        std::unordered_set<std::string>& local_seen,
        std::vector<Candidate>& local_out) {

        if (left == 0) {
            if (!passes_angle_no_zero_col(cur, polygon_n)) {
                return;
            }
            if (pingheng && !passes_balance(cur, polygon_n)) {
                return;
            }

            const std::string key = canonical_angle_key(cur, key_mode, polygon_n);
            if (local_seen.insert(key).second) {
                local_out.push_back(Candidate{ cur, key });
            }
            return;
        }

        const int n = static_cast<int>(rows.size());
        for (int i = start; i <= n - left; ++i) {
            set_row(cur, depth, rows[static_cast<std::size_t>(i)]);
            const Mat pref = prefix_rows(cur, depth + 1);
            if (rank_mat(pref) == depth + 1) {
                const int rows_left_after = left - 1;
                if (zero_angle_columns(pref, polygon_n) > vertex_order * rows_left_after) {
                    continue;
                }
                if (pingheng) {
                    Mat coefficients = first_cols(pref, polygon_n);
                    const std::vector<int> ones(static_cast<std::size_t>(polygon_n), 1);
                    if (rank_mat(append_one_row(coefficients, ones)) > target_rank) {
                        continue;
                    }
                }
                enumerate_angle_rec(rows, i + 1, left - 1, depth + 1, cur,
                    key_mode, pingheng, polygon_n, vertex_order, target_rank,
                    local_seen, local_out);
            }
        }
    }

    int zero_angle_columns(const Mat& a, int polygon_n) {
        int zeros = 0;
        for (int c = 0; c < polygon_n; ++c) {
            bool any = false;
            for (int r = 0; r < a.rows; ++r) {
                if (a.at(r, c) != 0) {
                    any = true;
                    break;
                }
            }
            if (!any) {
                ++zeros;
            }
        }
        return zeros;
    }

    std::vector<Mat> generate_angles_orbit_bfs(
        int target_rank,
        AngleKeyMode key_mode,
        bool pingheng,
        int polygon_n,
        int vertex_order,
        int num_threads) {

        if (target_rank <= 0 || target_rank > polygon_n) {
            return {};
        }

        const auto rows = make_angle_rows(polygon_n, vertex_order);
        const int nRows = static_cast<int>(rows.size());

        std::vector<Mat> states;
        states.emplace_back(0, polygon_n + 1);

        for (int depth = 0; depth < target_rank; ++depth) {
            if (states.empty()) {
                break;
            }

            const int next_dim = depth + 1;
            const int rows_left_after = target_rank - next_dim;
            const long long total = static_cast<long long>(states.size()) * nRows;
            const int buckets = choose_workers(num_threads, total);

            std::vector<std::vector<Candidate>> locals(static_cast<std::size_t>(buckets));
            std::vector<std::unordered_set<std::string>> local_seen(static_cast<std::size_t>(buckets));
            std::vector<std::unordered_set<std::string>> local_exact_seen(static_cast<std::size_t>(buckets));
            const std::vector<int> ones(static_cast<std::size_t>(polygon_n), 1);

            parallel_for_dynamic(0, total, buckets, [&](int tid, long long idx) {
                const std::size_t state_idx = static_cast<std::size_t>(idx / nRows);
                const std::size_t row_idx = static_cast<std::size_t>(idx % nRows);

                const Mat next = append_one_row(states[state_idx], rows[row_idx]);
                if (rank_mat(next) != next_dim) {
                    return;
                }

                if (zero_angle_columns(next, polygon_n) > vertex_order * rows_left_after) {
                    return;
                }

                if (pingheng) {
                    Mat angle_coefficients = first_cols(next, polygon_n);
                    const Mat with_ones = append_one_row(angle_coefficients, ones);
                    if (rank_mat(with_ones) > target_rank) {
                        return;
                    }
                }

                if (next_dim == target_rank) {
                    if (!passes_angle_no_zero_col(next, polygon_n)) {
                        return;
                    }
                    if (pingheng && !passes_balance(next, polygon_n)) {
                        return;
                    }
                }

                const std::string direct_key = rref_key(next);
                if (!local_exact_seen[static_cast<std::size_t>(tid)].insert(direct_key).second) {
                    return;
                }

                const std::string key = canonical_angle_key(next, key_mode, polygon_n);
                if (local_seen[static_cast<std::size_t>(tid)].insert(key).second) {
                    locals[static_cast<std::size_t>(tid)].push_back(Candidate{ next, key });
                }
                });

            states = merge_candidates(locals);
            mexPrintf(">>> C++: angle-only BFS rank %d representatives = %llu\n",
                next_dim, static_cast<unsigned long long>(states.size()));
            mexEvalString("drawnow;");
        }

        return states;
    }

    std::vector<Mat> generate_full_rank_angle_representative(int polygon_n, int vertex_order) {
        Mat a(polygon_n, polygon_n + 1);
        for (int r = 0; r < polygon_n; ++r) {
            a.at(r, r) = vertex_order;
            a.at(r, polygon_n) = 2;
        }
        return std::vector<Mat>{a};
    }

    std::vector<Mat> generate_angles(
        int target_rank,
        AngleKeyMode key_mode,
        bool pingheng,
        int polygon_n,
        int vertex_order,
        int num_threads) {

        if (target_rank <= 0 || target_rank > polygon_n) {
            return {};
        }

        if (target_rank == polygon_n) {
            return generate_full_rank_angle_representative(polygon_n, vertex_order);
        }

        if (key_mode != AngleKeyMode::Exact) {
            return generate_angles_orbit_bfs(
                target_rank, key_mode, pingheng, polygon_n, vertex_order, num_threads);
        }

        const auto rows = make_angle_rows(polygon_n, vertex_order);
        const int n = static_cast<int>(rows.size());
        const int k = target_rank;
        if (k > n) {
            return {};
        }

        const long long work_items = static_cast<long long>(n - k + 1);
        const int buckets = choose_workers(num_threads, work_items);
        std::vector<std::vector<Candidate>> locals(static_cast<std::size_t>(buckets));
        std::vector<std::unordered_set<std::string>> local_seen(static_cast<std::size_t>(buckets));

        parallel_for_dynamic(0, work_items, buckets, [&](int tid, long long first_ll) {
            const int first = static_cast<int>(first_ll);
            Mat cur(k, polygon_n + 1);
            set_row(cur, 0, rows[static_cast<std::size_t>(first)]);
            enumerate_angle_rec(rows, first + 1, k - 1, 1, cur,
                key_mode, pingheng, polygon_n, vertex_order, target_rank,
                local_seen[static_cast<std::size_t>(tid)],
                locals[static_cast<std::size_t>(tid)]);
            });

        return merge_candidates(locals);
    }

    bool contains_rowspace(const Mat& candidate, const Mat& known) {
        if (known.rows == 0) {
            return true;
        }
        const Mat stacked = append_matrix(candidate, known);
        return rank_mat(stacked) == candidate.rows;
    }

    std::vector<PairCandidate> merge_pairs(const std::vector<std::vector<PairCandidate>>& locals) {
        std::unordered_set<std::string> seen;
        std::vector<PairCandidate> out;
        for (const auto& bucket : locals) {
            for (const PairCandidate& p : bucket) {
                if (seen.insert(p.key).second) {
                    out.push_back(p);
                }
            }
        }
        return out;
    }

    std::vector<std::vector<std::string>> make_edge_orbit_keys(
        const std::vector<Mat>& edges,
        int polygon_n,
        int num_threads) {

        std::vector<std::vector<std::string>> keys(
            edges.size(), std::vector<std::string>(static_cast<std::size_t>(2 * polygon_n)));
        const int workers = choose_workers(num_threads, static_cast<long long>(edges.size()));
        parallel_for_dynamic(0, static_cast<long long>(edges.size()), workers, [&](int, long long i_ll) {
            const std::size_t i = static_cast<std::size_t>(i_ll);
            for (int t = 0; t < 2 * polygon_n; ++t) {
                keys[i][static_cast<std::size_t>(t)] = rref_key(
                    permute_edge(edges[i], edge_dihedral_perm(t, polygon_n)));
            }
            });
        return keys;
    }

    std::vector<std::vector<std::string>> make_joint_angle_orbit_keys(
        const std::vector<Mat>& angles,
        int polygon_n,
        int num_threads) {

        std::vector<std::vector<std::string>> keys(
            angles.size(), std::vector<std::string>(static_cast<std::size_t>(2 * polygon_n)));
        const int workers = choose_workers(num_threads, static_cast<long long>(angles.size()));
        parallel_for_dynamic(0, static_cast<long long>(angles.size()), workers, [&](int, long long i_ll) {
            const std::size_t i = static_cast<std::size_t>(i_ll);
            for (int t = 0; t < 2 * polygon_n; ++t) {
                keys[i][static_cast<std::size_t>(t)] = rref_key(
                    permute_angle(angles[i], angle_dihedral_perm(t, polygon_n)));
            }
            });
        return keys;
    }

    std::string canonical_pair_from_orbit_keys(
        const std::vector<std::string>& edge_keys,
        const std::vector<std::string>& angle_keys) {

        std::string best;
        bool has_best = false;
        for (std::size_t t = 0; t < edge_keys.size(); ++t) {
            const std::string key = pair_key_from_exact_keys(edge_keys[t], angle_keys[t]);
            if (!has_best || key < best) {
                best = key;
                has_best = true;
            }
        }
        return best;
    }

    std::vector<PairCandidate> generate_pairs(
        const std::vector<Mat>& edges,
        const std::vector<Mat>& angles,
        const std::vector<KnownPair>& known,
        int polygon_n,
        int num_threads) {

        const std::size_t nE = edges.size();
        const std::size_t nA = angles.size();
        const std::size_t nK = known.size();

        if (nE == 0 || nA == 0) {
            return {};
        }

        // RREFs of transformed matrices depend only on the individual edge or
        // angle candidate.  Precomputing them avoids 4*n RREF computations for
        // every element of the Cartesian product.
        const auto edge_orbit_keys = make_edge_orbit_keys(edges, polygon_n, num_threads);
        const auto angle_orbit_keys = make_joint_angle_orbit_keys(angles, polygon_n, num_threads);

        std::vector<std::vector<unsigned char>> e_has(nE, std::vector<unsigned char>(nK, 0));
        std::vector<std::vector<unsigned char>> a_has(nA, std::vector<unsigned char>(nK, 0));

        if (nK > 0) {
            const int e_workers = choose_workers(num_threads, static_cast<long long>(nE));
            parallel_for_dynamic(0, static_cast<long long>(nE), e_workers, [&](int, long long i_ll) {
                const std::size_t i = static_cast<std::size_t>(i_ll);
                for (std::size_t k = 0; k < nK; ++k) {
                    e_has[i][k] = contains_rowspace(edges[i], known[k].edge) ? 1 : 0;
                }
                });

            const int a_workers = choose_workers(num_threads, static_cast<long long>(nA));
            parallel_for_dynamic(0, static_cast<long long>(nA), a_workers, [&](int, long long i_ll) {
                const std::size_t i = static_cast<std::size_t>(i_ll);
                for (std::size_t k = 0; k < nK; ++k) {
                    a_has[i][k] = contains_rowspace(angles[i], known[k].angle) ? 1 : 0;
                }
                });
        }

        const long long total = static_cast<long long>(nE) * static_cast<long long>(nA);
        const int buckets = choose_workers(num_threads, total);
        std::vector<std::vector<PairCandidate>> locals(static_cast<std::size_t>(buckets));
        std::vector<std::unordered_set<std::string>> local_seen(static_cast<std::size_t>(buckets));

        parallel_for_dynamic(0, total, buckets, [&](int tid, long long idx) {
            const std::size_t ei = static_cast<std::size_t>(idx / static_cast<long long>(nA));
            const std::size_t ai = static_cast<std::size_t>(idx % static_cast<long long>(nA));

            bool degenerate = false;
            for (std::size_t k = 0; k < nK; ++k) {
                if (e_has[ei][k] && a_has[ai][k]) {
                    degenerate = true;
                    break;
                }
            }
            if (degenerate) {
                return;
            }

            const std::string key = canonical_pair_from_orbit_keys(
                edge_orbit_keys[ei], angle_orbit_keys[ai]);
            if (local_seen[static_cast<std::size_t>(tid)].insert(key).second) {
                locals[static_cast<std::size_t>(tid)].push_back(PairCandidate{ edges[ei], angles[ai], key });
            }
            });

        return merge_pairs(locals);
    }

    int scalar_int(const mxArray* a, const char* name) {
        if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) {
            mexErrMsgIdAndTxt("all_basis_generator:BadInput", "%s must be a real scalar double.", name);
        }
        const double value = mxGetScalar(a);
        if (!std::isfinite(value) || value != std::floor(value) ||
            value < static_cast<double>(std::numeric_limits<int>::min()) ||
            value > static_cast<double>(std::numeric_limits<int>::max())) {
            mexErrMsgIdAndTxt("all_basis_generator:BadInput", "%s must be a finite integer-valued scalar.", name);
        }
        return static_cast<int>(value);
    }

    int checked_int_value(double value, const char* name) {
        if (!std::isfinite(value) || value != std::floor(value) ||
            value < static_cast<double>(std::numeric_limits<int>::min()) ||
            value > static_cast<double>(std::numeric_limits<int>::max())) {
            mexErrMsgIdAndTxt(
                "all_basis_generator:BadKnownPool",
                "%s must contain finite integer-valued entries representable as C++ int.",
                name);
        }
        return static_cast<int>(value);
    }

    std::vector<KnownPair> parse_known(
        const mxArray* eArr,
        const mxArray* aArr,
        const mxArray* eRowsArr,
        const mxArray* aRowsArr,
        int polygon_n) {

        if (!mxIsDouble(eArr) || !mxIsDouble(aArr) || !mxIsDouble(eRowsArr) || !mxIsDouble(aRowsArr)) {
            mexErrMsgIdAndTxt("all_basis_generator:BadKnownPool", "Known_E, Known_A and row-count arrays must be double arrays.");
        }

        const mwSize* ed = mxGetDimensions(eArr);
        const mwSize* ad = mxGetDimensions(aArr);
        const mwSize e_nd = mxGetNumberOfDimensions(eArr);
        const mwSize a_nd = mxGetNumberOfDimensions(aArr);

        if (e_nd < 2 || a_nd < 2 || ed[1] != static_cast<mwSize>(polygon_n) ||
            ad[1] != static_cast<mwSize>(polygon_n + 1)) {
            mexErrMsgIdAndTxt("all_basis_generator:BadKnownPool",
                "Known_E must be maxRows x n x N and Known_A must be maxRows x (n+1) x N.");
        }

        const mwSize nE = (e_nd >= 3) ? ed[2] : 1;
        const mwSize nA = (a_nd >= 3) ? ad[2] : 1;
        const mwSize nR = std::min(mxGetNumberOfElements(eRowsArr), mxGetNumberOfElements(aRowsArr));
        const mwSize n = std::min(std::min(nE, nA), nR);

        const double* ep = mxGetPr(eArr);
        const double* ap = mxGetPr(aArr);
        const double* erp = mxGetPr(eRowsArr);
        const double* arp = mxGetPr(aRowsArr);

        std::vector<KnownPair> out;
        out.reserve(static_cast<std::size_t>(n));

        for (mwSize k = 0; k < n; ++k) {
            const int rE = checked_int_value(erp[k], "Known_E_rows");
            const int rA = checked_int_value(arp[k], "Known_A_rows");
            if (rE <= 0 || rA <= 0) {
                continue;
            }
            if (static_cast<mwSize>(rE) > ed[0] || static_cast<mwSize>(rA) > ad[0]) {
                mexErrMsgIdAndTxt("all_basis_generator:BadKnownPool", "Known row count exceeds the allocated known_pool array size.");
            }

            Mat e(rE, polygon_n);
            Mat a(rA, polygon_n + 1);

            for (int r = 0; r < rE; ++r) {
                for (int c = 0; c < polygon_n; ++c) {
                    const mwSize idx = static_cast<mwSize>(r) + ed[0] *
                        (static_cast<mwSize>(c) + static_cast<mwSize>(polygon_n) * k);
                    e.at(r, c) = checked_int_value(ep[idx], "Known_E");
                }
            }
            for (int r = 0; r < rA; ++r) {
                for (int c = 0; c < polygon_n + 1; ++c) {
                    const mwSize idx = static_cast<mwSize>(r) + ad[0] *
                        (static_cast<mwSize>(c) + static_cast<mwSize>(polygon_n + 1) * k);
                    a.at(r, c) = checked_int_value(ap[idx], "Known_A");
                }
            }

            out.push_back(KnownPair{ e, a });
        }

        return out;
    }

    mxArray* mats_to_mx(const std::vector<Mat>& mats, int rows, int cols) {
        if (mats.empty()) {
            return mxCreateDoubleMatrix(0, static_cast<mwSize>(cols), mxREAL);
        }

        const mwSize dims[3] = {
            static_cast<mwSize>(rows),
            static_cast<mwSize>(cols),
            static_cast<mwSize>(mats.size())
        };
        mxArray* out = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        double* p = mxGetPr(out);

        for (std::size_t k = 0; k < mats.size(); ++k) {
            const Mat& m = mats[k];
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    const mwSize idx = static_cast<mwSize>(r) +
                        static_cast<mwSize>(rows) * (static_cast<mwSize>(c) + static_cast<mwSize>(cols) * static_cast<mwSize>(k));
                    p[idx] = static_cast<double>(m.at(r, c));
                }
            }
        }
        return out;
    }

    mxArray* pair_edges_to_mx(const std::vector<PairCandidate>& pairs, int rows, int polygon_n) {
        if (pairs.empty()) {
            return mxCreateDoubleMatrix(0, static_cast<mwSize>(polygon_n), mxREAL);
        }

        const mwSize dims[3] = {
            static_cast<mwSize>(rows),
            static_cast<mwSize>(polygon_n),
            static_cast<mwSize>(pairs.size())
        };
        mxArray* out = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        double* p = mxGetPr(out);
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < polygon_n; ++c) {
                    const mwSize idx = static_cast<mwSize>(r) +
                        static_cast<mwSize>(rows) * (static_cast<mwSize>(c) +
                            static_cast<mwSize>(polygon_n) * static_cast<mwSize>(k));
                    p[idx] = static_cast<double>(pairs[k].edge.at(r, c));
                }
            }
        }
        return out;
    }

    mxArray* pair_angles_to_mx(const std::vector<PairCandidate>& pairs, int rows, int polygon_n) {
        if (pairs.empty()) {
            return mxCreateDoubleMatrix(0, static_cast<mwSize>(polygon_n + 1), mxREAL);
        }

        const mwSize dims[3] = {
            static_cast<mwSize>(rows),
            static_cast<mwSize>(polygon_n + 1),
            static_cast<mwSize>(pairs.size())
        };
        mxArray* out = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        double* p = mxGetPr(out);
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < polygon_n + 1; ++c) {
                    const mwSize idx = static_cast<mwSize>(r) +
                        static_cast<mwSize>(rows) * (static_cast<mwSize>(c) +
                            static_cast<mwSize>(polygon_n + 1) * static_cast<mwSize>(k));
                    p[idx] = static_cast<double>(pairs[k].angle.at(r, c));
                }
            }
        }
        return out;
    }

} // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 10 && nrhs != 11) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput",
            "Usage: [E_results,A_results] = all_basis_generator(Target_RE,Target_RA,pingheng,Num_Threads,n,jieshu[,pu],Known_E,Known_A,Known_E_rows,Known_A_rows)");
    }
    if (nlhs < 1 || nlhs > 2) {
        mexErrMsgIdAndTxt("all_basis_generator:BadOutput", "One or two outputs are required.");
    }

    const int targetRE = scalar_int(prhs[0], "Target_RE");
    const int targetRA = scalar_int(prhs[1], "Target_RA");
    const bool pingheng = scalar_int(prhs[2], "pingheng") != 0;
    const int numThreads = std::max(1, scalar_int(prhs[3], "Num_Threads"));
    const int polygonN = scalar_int(prhs[4], "n");
    const int vertexOrder = scalar_int(prhs[5], "jieshu");
    const bool hasPu = nrhs == 11;
    const int pu = hasPu ? scalar_int(prhs[6], "pu") : 0;
    const int knownOffset = hasPu ? 7 : 6;

    if (targetRE < 0 || targetRA < 0) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "Target ranks must be non-negative.");
    }
    if (polygonN < 3) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "n must be at least 3.");
    }
    if (vertexOrder < 1) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "jieshu must be a positive integer.");
    }
    if (pu != 0 && pu != 1) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "pu must be 0 or 1.");
    }
    if (targetRE > polygonN - 1) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "Target_RE cannot exceed n-1.");
    }
    if (targetRA > polygonN) {
        mexErrMsgIdAndTxt("all_basis_generator:BadInput", "Target_RA cannot exceed n.");
    }
    if (targetRE == 0 && targetRA == 0) {
        plhs[0] = mats_to_mx({}, 0, polygonN);
        if (nlhs > 1) {
            plhs[1] = mats_to_mx({}, 0, polygonN + 1);
        }
        return;
    }

    const std::vector<KnownPair> baseKnown = parse_known(
        prhs[knownOffset],
        prhs[knownOffset + 1],
        prhs[knownOffset + 2],
        prhs[knownOffset + 3],
        polygonN);

    mexPrintf(">>> C++: generating candidates for n=%d, jieshu=%d, pu=%d...\n",
        polygonN, vertexOrder, pu);
    mexEvalString("drawnow;");

    if (targetRE == 0) {
        const AngleKeyMode angleOnlyMode = pu == 0
            ? AngleKeyMode::Symmetric
            : AngleKeyMode::Dihedral;
        const std::vector<Mat> angles = generate_angles(
            targetRA, angleOnlyMode, pingheng, polygonN, vertexOrder, numThreads);
        mexPrintf(">>> C++: angle-only result count = %llu\n", static_cast<unsigned long long>(angles.size()));
        plhs[0] = mats_to_mx(std::vector<Mat>{}, 0, polygonN);
        if (nlhs > 1) {
            plhs[1] = mats_to_mx(angles, targetRA, polygonN + 1);
        }
        return;
    }

    if (targetRA == 0) {
        const std::vector<Mat> edges = generate_edges(
            targetRE, vertexOrder == 3, true, polygonN, numThreads);
        mexPrintf(">>> C++: edge-only result count = %llu\n", static_cast<unsigned long long>(edges.size()));
        plhs[0] = mats_to_mx(edges, targetRE, polygonN);
        if (nlhs > 1) {
            plhs[1] = mats_to_mx(std::vector<Mat>{}, 0, polygonN + 1);
        }
        return;
    }

    const std::vector<Mat> edges = generate_edges(
        targetRE, vertexOrder == 3, true, polygonN, numThreads);
    mexPrintf(">>> C++: edge D%d representatives before joint pairing = %llu\n",
        polygonN, static_cast<unsigned long long>(edges.size()));
    mexEvalString("drawnow;");

    const std::vector<Mat> angles = generate_angles(
        targetRA, AngleKeyMode::Exact, pingheng, polygonN, vertexOrder, numThreads);
    mexPrintf(">>> C++: angle representatives = %llu\n", static_cast<unsigned long long>(angles.size()));
    mexEvalString("drawnow;");

    const std::vector<KnownPair>& known = baseKnown;
    mexPrintf(">>> C++: known_pool degeneracy entries = %llu\n", static_cast<unsigned long long>(known.size()));
    mexEvalString("drawnow;");

    const std::vector<PairCandidate> pairs = generate_pairs(edges, angles, known, polygonN, numThreads);
    mexPrintf(">>> C++: joint result count = %llu\n", static_cast<unsigned long long>(pairs.size()));
    mexEvalString("drawnow;");

    plhs[0] = pair_edges_to_mx(pairs, targetRE, polygonN);
    if (nlhs > 1) {
        plhs[1] = pair_angles_to_mx(pairs, targetRA, polygonN);
    }
}*/
