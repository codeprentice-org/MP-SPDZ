/*
 * Rep4.h
 *
 */

#ifndef PROTOCOLS_REP4_H_
#define PROTOCOLS_REP4_H_

#include "Replicated.h"

template<class T>
class Rep4 : public ProtocolBase<T>
{
    friend class Rep4RingPrep<T>;

    typedef typename T::open_type open_type;

    array<octetStream, 2> os;
    array<array<Hash, 4>, 4> send_hashes, receive_hashes;

    array<vector<open_type>, 5> add_shares;
    vector<int> bit_lengths;

    class ResTuple
    {
    public:
        T res;
        open_type r;
    };

    PointerVector<ResTuple> results;

    int my_num;

    array<open_type, 5> get_addshares(const T& x, const T& y);

    void reset_joint_input(int n_inputs);
    void prepare_joint_input(int sender, int backup, int receiver,
            int outsider, vector<open_type>& inputs);
    void finalize_joint_input(int sender, int backup, int receiver,
            int outsider);

    int get_player(int offset);

public:
    array<ElementPRNG<typename T::open_type>, 3> rep_prngs;
    Player& P;

    Rep4(Player& P);

    void init_mul(SubProcessor<T>* proc = 0);
    void init_mul(Preprocessing<T>& prep, typename T::MAC_Check& MC);
    typename T::clear prepare_mul(const T& x, const T& y, int n = -1);
    void exchange();
    T finalize_mul(int n = -1);
    void check();

    void init_dotprod(SubProcessor<T>* proc);
    void prepare_dotprod(const T& x, const T& y);
    void next_dotprod();
    T finalize_dotprod(int length);

    T get_random();
    void randoms(T& res, int n_bits);

    void trunc_pr(const vector<int>& regs, int size, SubProcessor<T>& proc);

    int get_n_relevant_players() { return 2; }
};

#endif /* PROTOCOLS_REP4_H_ */
