/*
 * Rep4.cpp
 *
 */

#include "Rep4.h"
#include "Processor/TruncPrTuple.h"

template<class T>
Rep4<T>::Rep4(Player& P) :
        my_num(P.my_num()), P(P)
{
    assert(P.num_players() == 4);

    rep_prngs[0].ReSeed();
    for (int i = 1; i < 3; i++)
    {
        octetStream os;
        os.append(rep_prngs[0].get_seed(), SEED_SIZE);
        P.pass_around(os, -i);
        rep_prngs[i].SetSeed(os.get_data());
    }
}

template<class T>
void Rep4<T>::init_mul(SubProcessor<T>*)
{
    for (auto& x : add_shares)
        x.clear();
    bit_lengths.clear();
}

template<class T>
void Rep4<T>::init_mul(Preprocessing<T>&, typename T::MAC_Check&)
{
    init_mul();
}

template<class T>
void Rep4<T>::reset_joint_input(int n_inputs)
{
    results.clear();
    results.resize(n_inputs);
    bit_lengths.clear();
    bit_lengths.resize(n_inputs, -1);
}

template<class T>
void Rep4<T>::prepare_joint_input(int sender, int backup, int receiver,
        int outsider, vector<open_type>& inputs)
{
    if (P.my_num() != receiver)
    {
        int index = P.get_offset(receiver) - 1;
        for (auto& x : results)
        {
            x.r = rep_prngs[index].get();
            x.res[index] += x.r;
        }

        if (P.my_num() == sender or P.my_num() == backup)
        {
            int offset = P.get_offset(outsider) - 1;
            for (size_t i = 0; i < results.size(); i++)
            {
                auto& input = inputs[i];
                input -= results[i].r;
                results[i].res[offset] += input;
            }
        }
    }

    if (P.my_num() == backup)
    {
        send_hashes[sender][receiver].update(inputs);
    }

    if (sender == P.my_num())
    {
        assert(inputs.size() == bit_lengths.size());
        switch (P.get_offset(backup))
        {
        case 2:
            for (size_t i = 0; i < inputs.size(); i++)
                inputs[i].pack(os[1], bit_lengths[i]);
            break;
        case 1:
            for (size_t i = 0; i < inputs.size(); i++)
                inputs[i].pack(os[0], bit_lengths[i]);
            break;
        default:
            throw not_implemented();
        }
    }
}

template<class T>
void Rep4<T>::finalize_joint_input(int sender, int backup, int receiver,
        int)
{
    if (P.my_num() == receiver)
    {
        assert(results.size() == bit_lengths.size());
        T res;
        switch (P.get_offset(backup))
        {
        case 2:
            receive_hashes[sender][backup].update(os[0].get_data_ptr(),
                    results.size() * open_type::size());
            for (size_t i = 0; i < results.size(); i++)
            {
                auto& x = results[i];
                res[2].unpack(os[0], bit_lengths[i]);
                x.res[2] += res[2];
            }
            break;
        default:
            receive_hashes[sender][backup].update(os[1].get_data_ptr(),
                    results.size() * open_type::size());
            for (size_t i = 0; i < results.size(); i++)
            {
                auto& x = results[i];
                res[1].unpack(os[1], bit_lengths[i]);
                x.res[1] += res[1];
            }
            break;
        }
    }
}

template<class T>
int Rep4<T>::get_player(int offset)
{
    return (my_num + offset) & 3;
}

template<class T>
typename T::clear Rep4<T>::prepare_mul(const T& x, const T& y, int n_bits)
{
    auto a = get_addshares(x, y);
    for (int i = 0; i < 5; i++)
        add_shares[i].push_back(a[i]);
    bit_lengths.push_back(n_bits);
    return {};
}

template<class T>
array<typename T::open_type, 5> Rep4<T>::get_addshares(const T& x, const T& y)
{
    array<open_type, 5> res;
    for (int i = 0; i < 2; i++)
        res[get_player(i - 1)] =
                (x[i] + x[i + 1]) * y[i] + x[i] * y[i + 1];
    res[4] = x[0] * y[2] + x[2] * y[0];
    return res;
}

template<class T>
void Rep4<T>::init_dotprod(SubProcessor<T>*)
{
    init_mul();
    next_dotprod();
}

template<class T>
void Rep4<T>::prepare_dotprod(const T& x, const T& y)
{
    auto a = get_addshares(x, y);
    for (int i = 0; i < 5; i++)
        add_shares[i].back() += a[i];
}

template<class T>
void Rep4<T>::next_dotprod()
{
    for (auto& a : add_shares)
        a.push_back({});
    bit_lengths.push_back(-1);
}

template<class T>
void Rep4<T>::exchange()
{
    for (auto& o : os)
        o.reset_write_head();
    auto& a = add_shares;
    results.clear();
    results.resize(a[4].size());
    prepare_joint_input(0, 1, 3, 2, a[0]);
    prepare_joint_input(1, 2, 0, 3, a[1]);
    prepare_joint_input(2, 3, 1, 0, a[2]);
    prepare_joint_input(3, 0, 2, 1, a[3]);
    prepare_joint_input(0, 2, 3, 1, a[4]);
    prepare_joint_input(1, 3, 2, 0, a[4]);
    P.pass_around(os[0], -1);
    if (P.my_num() < 2)
        P.send_to(3 - P.my_num(), os[1], true);
    else
        P.receive_player(3 - P.my_num(), os[1], true);
    finalize_joint_input(0, 1, 3, 2);
    finalize_joint_input(1, 2, 0, 3);
    finalize_joint_input(2, 3, 1, 0);
    finalize_joint_input(3, 0, 2, 1);
    finalize_joint_input(0, 2, 3, 1);
    finalize_joint_input(1, 3, 2, 0);
}

template<class T>
T Rep4<T>::finalize_mul(int)
{
    this->counter++;
    return results.next().res;
}

template<class T>
T Rep4<T>::finalize_dotprod(int)
{
    this->counter++;
    return finalize_mul();
}

template<class T>
void Rep4<T>::check()
{
    for (int i = 1; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            octetStream os;
            send_hashes[j][P.get_player(i)].final(os);
            P.pass_around(os, i);
            if (receive_hashes[j][P.get_player(-i)].final() != os)
                throw runtime_error(
                        "hash mismatch for sender " + to_string(j)
                                + " and backup " + to_string(P.get_player(-i)));
        }
    }
}

template<class T>
T Rep4<T>::get_random()
{
    T res;
    for (int i = 0; i < 3; i++)
        res[i].randomize(rep_prngs[i]);
    return res;
}

template<class T>
void Rep4<T>::randoms(T& res, int n_bits)
{
    for (int i = 0; i < 3; i++)
        res[i].randomize_part(rep_prngs[i], n_bits);
}

template<class T>
void Rep4<T>::trunc_pr(const vector<int>& regs, int size,
		SubProcessor<T>& proc)
{
    assert(regs.size() % 4 == 0);
    typedef typename T::open_type open_type;

    vector<TruncPrTupleWithGap<open_type>> infos;
    for (size_t i = 0; i < regs.size(); i += 4)
        infos.push_back({regs, i});

    PointerVector<T> rs(size * infos.size());
    for (int i = 2; i < 4; i++)
    {
        int index = P.get_offset(i) - 1;
        if (index >= 0)
            for (auto& r : rs)
                r[index].randomize(rep_prngs[index]);
    }

    vector<T> cs;
    cs.reserve(rs.size());
    for (auto& info : infos)
    {
        for (int j = 0; j < size; j++)
            cs.push_back(proc.get_S_ref(info.source_base + j) + rs.next());
    }

    octetStream c_os;
    vector<open_type> eval_inputs;
    if (P.my_num() < 2)
    {
        if (P.my_num() == 0)
            for (auto& c : cs)
                (c[1] + c[2]).pack(c_os);
        else
            for (auto& c : cs)
                (c[1] + c[0]).pack(c_os);
        P.send_to(2 + P.my_num(), c_os, true);
        P.send_to(3 - P.my_num(), c_os.hash(), true);
    }
    else
    {
        P.receive_player(P.my_num() - 2, c_os, true);
        octetStream hash;
        P.receive_player(3 - P.my_num(), hash, true);
        if (hash != c_os.hash())
            throw runtime_error("hash mismatch in joint message passing");
        PointerVector<open_type> open_cs;
        if (P.my_num() == 2)
            for (auto& c : cs)
                open_cs.push_back(c_os.get<open_type>() + c[1] + c[2]);
        else
            for (auto& c : cs)
                open_cs.push_back(c_os.get<open_type>() + c[1] + c[0]);
        for (auto& info : infos)
            for (int j = 0; j < size; j++)
            {
                auto c = open_cs.next();
                auto c_prime = info.upper(c);
                if (not info.big_gap())
                {
                    auto c_msb = info.msb(c);
                    eval_inputs.push_back(c_msb);
                }
                eval_inputs.push_back(c_prime);
            }
    }

    PointerVector<open_type> inputs;
    bool generate = proc.P.my_num() < 2;
    if (generate)
    {
        inputs.reserve(2 * rs.size());
        rs.reset();
        for (auto& info : infos)
            for (int j = 0; j < size; j++)
            {
                auto& r = rs.next();
                if (not info.big_gap())
                    inputs.push_back(info.msb(r.sum()));
                inputs.push_back(info.upper(r.sum()));
            }
    }

    for (auto& o : os)
        o.clear();
    size_t n_inputs = max(inputs.size(), eval_inputs.size());
    reset_joint_input(n_inputs);
    prepare_joint_input(0, 1, 3, 2, inputs);
    if (P.my_num() == 0)
        P.send_to(3, os[0], true);
    else if (P.my_num() == 3)
        P.receive_player(0, os[0], true);
    finalize_joint_input(0, 1, 3, 2);
    PointerVector<T> gen_results;
    for (auto& x : results)
        gen_results.push_back(x.res);

    for (auto& o : os)
        o.clear();
    reset_joint_input(n_inputs);
    prepare_joint_input(2, 3, 1, 0, eval_inputs);
    if (P.my_num() == 2)
        P.send_to(1, os[0], true);
    else if (P.my_num() == 1)
        P.receive_player(2, os[0], true);
    finalize_joint_input(2, 3, 1, 0);
    PointerVector<T> eval_results;
    for (auto& x : results)
        eval_results.push_back(x.res);

    init_mul();
    for (auto& info : infos)
        for (int j = 0; j < size; j++)
        {
            if (not info.big_gap())
                prepare_mul(gen_results.next(), eval_results.next());
            gen_results.next();
            eval_results.next();
        }

    if (not add_shares[0].empty())
        exchange();

    eval_results.reset();
    gen_results.reset();

    for (auto& info : infos)
        for (int j = 0; j < size; j++)
        {
            if (info.big_gap())
                proc.get_S_ref(info.dest_base + j) = eval_results.next()
                        - gen_results.next();
            else
            {
                auto b = gen_results.next() + eval_results.next()
                        - 2 * finalize_mul();
                proc.get_S_ref(info.dest_base + j) = eval_results.next()
                            - gen_results.next() + (b << (info.k - info.m));
            }
        }
}
