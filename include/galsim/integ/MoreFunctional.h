/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#ifndef GalSim_MoreFunc_H
#define GalSim_MoreFunc_H

/**
 * @file MoreFunctional.h
 *
 * @brief Some additional functional operators that aren't in the standard library's
 *        "<functional>" but should be.
 */


#include <functional>

namespace galsim {
namespace integ {

    template <class Ret, class T>
    class MemberLess_t : public std::binary_function<const T&,const T&,bool> 
    {
    public:
        explicit MemberLess_t(Ret (T::*_f)() const) : f(_f) {}
        bool operator()(const T& t1, const T& t2) const 
        { return (t1.*f)() < (t2.*f)(); }
    private:
        Ret (T::*f)() const;
    };

    template <class Ret, class T>
    class PtrMemberLess_t : public std::binary_function<const T&,const T&,bool> 
    {
    public:
        explicit PtrMemberLess_t(Ret (T::*_f)() const) : f(_f) {}
        bool operator()(const T* t1, const T* t2) const 
        {
            return (t1->*f)() < (t2->*f)(); 
        }
    private:
        Ret (T::*f)() const;
    };

    template <class Ret, class T>
    inline MemberLess_t<Ret,T> MemberLess(Ret (T::*f)() const)
    { return MemberLess_t<Ret,T>(f); }

    template <class Ret, class T>
    inline PtrMemberLess_t<Ret,T> PtrMemberLess(Ret (T::*f)() const)
    { return PtrMemberLess_t<Ret,T>(f); }

    template <class Ret, class T>
    class MemberGreater_t : public std::binary_function<const T&,const T&,bool> 
    {
    public:
        explicit MemberGreater_t(Ret (T::*_f)() const) : f(_f) {}
        bool operator()(const T& t1, const T& t2) const 
        { return (t1.*f)() < (t2.*f)(); }
    private:
        Ret (T::*f)() const;
    };

    template <class Ret, class T>
    class PtrMemberGreater_t : public std::binary_function<const T&,const T&,bool> 
    {
    public:
        explicit PtrMemberGreater_t(Ret (T::*_f)() const) : f(_f) {}
        bool operator()(const T* t1, const T* t2) const 
        { return (t1->*f)() < (t2->*f)(); }
    private:
        Ret (T::*f)() const;
    };

    template <class Ret, class T>
    inline MemberGreater_t<Ret,T> MemberGreater(Ret (T::*f)() const)
    { return MemberGreater_t<Ret,T>(f); }

    template <class Ret, class T>
    inline PtrMemberGreater_t<Ret,T> PtrMemberGreater(Ret (T::*f)() const)
    { return PtrMemberGreater_t<Ret,T>(f); }

    template <class Ret, class T>
    class MemberEqual_t : public std::binary_function<const T&,const T&,bool> 
    {
    public:
        explicit MemberEqual_t(Ret (T::*_f)() const) : f(_f) {}
        bool operator()(const T& t1, const T& t2) const 
        { return (t1.*f)() < (t2.*f)(); }
    private:
        Ret (T::*f)() const;
    };

    template <class Ret, class T>
    class PtrMemberEqual_t : public std::binary_function<const T&,const T&,bool> 
    {
    public:
        explicit PtrMemberEqual_t(Ret (T::*_f)() const) : f(_f) {}
        bool operator()(const T* t1, const T* t2) const 
        { return (t1->*f)() < (t2->*f)(); }
    private:
        Ret (T::*f)() const;
    };

    template <class Ret, class T>
    inline MemberEqual_t<Ret,T> MemberEqual(Ret (T::*f)() const)
    { return MemberEqual_t<Ret,T>(f); }

    template <class Ret, class T>
    inline PtrMemberEqual_t<Ret,T> PtrMemberEqual(Ret (T::*f)() const)
    { return PtrMemberEqual_t<Ret,T>(f); }

    template <class T>
    struct PtrLess : std::binary_function<T*,T*,bool> 
    {
        bool operator()(const T* p1, const T* p2) const 
        { return *p1<*p2; }
    };

    template <class T>
    struct PtrGreater : std::binary_function<T*,T*,bool> 
    {
        bool operator()(const T* p1, const T* p2) const 
        { return *p1>*p2; }
    };

    template <class T>
    struct PtrEqual : std::binary_function<T*,T*,bool> 
    {
        bool operator()(const T* p1, const T* p2) const 
        { return *p1==*p2; }
    };


    // Extend standard unary and binary functions to 
    // nullary, trinary, tetranary, pentanary, hexanary, heptanary, octanary 
    // functions

    template <class _Result>
    struct nullary_function 
    {
        typedef _Result result_type;
    };      

    template <class _Arg1, class _Arg2, class _Arg3, class _Result>
    struct trinary_function 
    {
        typedef _Arg1 firstof3_argument_type;
        typedef _Arg2 secondof3_argument_type;
        typedef _Arg3 thirdof3_argument_type;
        typedef _Result result_type;
    };      

    template <class _Arg1, class _Arg2, class _Arg3, class _Arg4, class _Result>
    struct tetranary_function 
    {
        typedef _Arg1 firstof4_argument_type;
        typedef _Arg2 secondof4_argument_type;
        typedef _Arg3 thirdof4_argument_type;
        typedef _Arg4 fourthof4_argument_type;
        typedef _Result result_type;
    };      

    template <class _Arg1, class _Arg2, class _Arg3, class _Arg4, 
              class _Arg5, class _Result>
    struct pentanary_function 
    {
        typedef _Arg1 firstof5_argument_type;
        typedef _Arg2 secondof5_argument_type;
        typedef _Arg3 thirdof5_argument_type;
        typedef _Arg4 fourthof5_argument_type;
        typedef _Arg5 fifthof5_argument_type;
        typedef _Result result_type;
    };      

    template <class _Arg1, class _Arg2, class _Arg3, class _Arg4, 
              class _Arg5, class _Arg6, class _Result>
    struct hexanary_function 
    {
        typedef _Arg1 firstof6_argument_type;
        typedef _Arg2 secondof6_argument_type;
        typedef _Arg3 thirdof6_argument_type;
        typedef _Arg4 fourthof6_argument_type;
        typedef _Arg5 fifthof6_argument_type;
        typedef _Arg6 sixthof6_argument_type;
        typedef _Result result_type;
    };      

    template <class _Arg1, class _Arg2, class _Arg3, class _Arg4, 
              class _Arg5, class _Arg6, class _Arg7, class _Result>
    struct heptanary_function 
    {
        typedef _Arg1 firstof7_argument_type;
        typedef _Arg2 secondof7_argument_type;
        typedef _Arg3 thirdof7_argument_type;
        typedef _Arg4 fourthof7_argument_type;
        typedef _Arg5 fifthof7_argument_type;
        typedef _Arg6 sixthof7_argument_type;
        typedef _Arg7 seventhof7_argument_type;
        typedef _Result result_type;
    };      

    template <class _Arg1, class _Arg2, class _Arg3, class _Arg4, 
              class _Arg5, class _Arg6, class _Arg7, class _Arg8, class _Result>
    struct octanary_function 
    {
        typedef _Arg1 firstof8_argument_type;
        typedef _Arg2 secondof8_argument_type;
        typedef _Arg3 thirdof8_argument_type;
        typedef _Arg4 fourthof8_argument_type;
        typedef _Arg5 fifthof8_argument_type;
        typedef _Arg6 sixthof8_argument_type;
        typedef _Arg7 seventhof8_argument_type;
        typedef _Arg8 eighthof8_argument_type;
        typedef _Result result_type;
    };      

    template <class _UF>
    class binder1_1 : public nullary_function<typename _UF::result_type> 
    {
    protected:
        const _UF& op;
        typename _UF::argument_type value;
    public:
        binder1_1(const _UF& __x,
                  typename _UF::argument_type __y)
            : op(__x), value(__y) {}
        typename _UF::result_type operator()() const 
        { return op(value); }
    };

    template <class _UF, class _Tp>
    inline binder1_1<_UF> bind11(const _UF& __oper, const _Tp& __x) 
    {
        typedef typename _UF::argument_type _Arg;
        return binder1_1<_UF>(__oper, static_cast<_Arg>(__x));
    }

    template <class _BF>
    class binder2_1 :
        public std::unary_function<typename _BF::second_argument_type,
        typename _BF::result_type> 
    {
    protected:
        _BF oper;
        typename _BF::first_argument_type value;
    public:
        binder2_1(const _BF& __oper, typename _BF::first_argument_type __val) : 
            oper(__oper), value(__val) {}

        typename _BF::result_type operator()(
            const typename _BF::second_argument_type& __x) const 
        { return oper(value, __x); }
    };

    template <class _BF, class _Tp>
    inline binder2_1<_BF> bind21(const _BF& __oper, const _Tp& __x) 
    {
        typedef typename _BF::first_argument_type _Arg;
        return binder2_1<_BF>(__oper, static_cast<_Arg>(__x));
    }

    template <class _BF>
    class binder2_2 : 
        public std::unary_function<typename _BF::first_argument_type,
        typename _BF::result_type> 
    {
    protected:
        _BF oper;
        typename _BF::second_argument_type value;
    public:
        binder2_2(const _BF& __oper, typename _BF::second_argument_type __val) :
            oper(__oper), value(__val) {}
        typename _BF::result_type operator()(
            const typename _BF::first_argument_type& __x) const 
        { return oper(__x, value); }
    };

    template <class _BF, class _Tp>
    inline binder2_2<_BF> bind22(const _BF& __oper, const _Tp& __x) 
    {
        typedef typename _BF::second_argument_type _Arg;
        return binder2_2<_BF>(__oper, static_cast<_Arg>(__x));
    }

    template <class _TF>
    class binder3_1 : 
        public std::binary_function<typename _TF::secondof3_argument_type,
        typename _TF::thirdof3_argument_type,
        typename _TF::result_type> 
    {
    protected:
        _TF oper;
        typename _TF::firstof3_argument_type value;
    public:
        binder3_1(const _TF& __oper, typename _TF::firstof3_argument_type __val) :
            oper(__oper), value(__val) {}
        typename _TF::result_type operator()(
            const typename _TF::secondof3_argument_type& __x1, 
            const typename _TF::thirdof3_argument_type& __x2) const 
        { return oper(value, __x1, __x2); }
    };

    template <class _TF, class _Tp>
    inline binder3_1<_TF> bind31(const _TF& __oper, const _Tp& __x) 
    {
        typedef typename _TF::firstof3_argument_type _Arg;
        return binder3_1<_TF>(__oper, static_cast<_Arg>(__x));
    }

    template <class _TF>
    class binder3_2 : 
        public std::binary_function<typename _TF::firstof3_argument_type,
        typename _TF::thirdof3_argument_type,
        typename _TF::result_type> 
    {
    protected:
        _TF oper;
        typename _TF::secondof3_argument_type value;
    public:
        binder3_2(const _TF& __oper, typename _TF::secondof3_argument_type __val) :
            oper(__oper), value(__val) {}
        typename _TF::result_type operator()(
            const typename _TF::firstof3_argument_type& __x1, 
            const typename _TF::thirdof3_argument_type& __x2) const 
        { return oper(__x1, value, __x2); }
    };

    template <class _TF, class _Tp>
    inline binder3_2<_TF> bind32(const _TF& __oper, const _Tp& __x) 
    {
        typedef typename _TF::secondof3_argument_type _Arg;
        return binder3_2<_TF>(__oper, static_cast<_Arg>(__x));
    }

    template <class _TF>
    class binder3_3 : 
        public std::binary_function<typename _TF::firstof3_argument_type,
        typename _TF::secondof3_argument_type,
        typename _TF::result_type> 
    {
    protected:
        _TF oper;
        typename _TF::thirdof3_argument_type value;
    public:
        binder3_3(const _TF& __oper, typename _TF::thirdof3_argument_type __val) :
            oper(__oper), value(__val) {}
        typename _TF::result_type operator()(
            const typename _TF::firstof3_argument_type& __x1, 
            const typename _TF::secondof3_argument_type& __x2) const 
        { return oper(__x1, __x2, value); }
    };

    template <class _TF, class _Tp>
    inline binder3_3<_TF> bind33(const _TF& __oper, const _Tp& __x) 
    {
        typedef typename _TF::thirdof3_argument_type _Arg;
        return binder3_3<_TF>(__oper, static_cast<_Arg>(__x));
    }

    template <class _Result>
    class pointer_to_nullary_function : public nullary_function<_Result> 
    {
    protected:
        _Result (*_M_ptr)();
    public:
        pointer_to_nullary_function() {}
        explicit pointer_to_nullary_function(_Result (*__x)()) : _M_ptr(__x) {}
        _Result operator()() const { return _M_ptr(); }
    };

    template <class _Result>
    inline pointer_to_nullary_function<_Result> ptr_fun(_Result (*__x)())
    {
        return pointer_to_nullary_function<_Result>(__x);
    }

    template <class _Arg1, class _Arg2, class _Arg3, class _Result>
    class pointer_to_trinary_function : 
        public trinary_function<_Arg1,_Arg2,_Arg3,_Result> 
    {
    protected:
        _Result (*_M_ptr)(_Arg1, _Arg2, _Arg3);
    public:
        pointer_to_trinary_function() {}
        explicit pointer_to_trinary_function(_Result (*__x)(_Arg1, _Arg2, _Arg3)) : 
            _M_ptr(__x) {}
        _Result operator()(_Arg1 __x1, _Arg2 __x2, _Arg3 __x3) const 
        { return _M_ptr(__x1, __x2, __x3); }
    };

    template <class _Arg1, class _Arg2, class _Arg3, class _Result>
    inline pointer_to_trinary_function<_Arg1,_Arg2,_Arg3,_Result> ptr_fun(
        _Result (*__x)(_Arg1, _Arg2, _Arg3)) 
    {
        return pointer_to_trinary_function<_Arg1,_Arg2,_Arg3,_Result>(__x);
    }

    template <class _Arg1, class _Arg2, class _Arg3, class _Arg4, class _Result>
    class pointer_to_tetranary_function : 
        public tetranary_function<_Arg1,_Arg2,_Arg3,_Arg4,_Result> 
    {
    protected:
        _Result (*_M_ptr)(_Arg1, _Arg2, _Arg3, _Arg4);
    public:
        pointer_to_tetranary_function() {}
        explicit pointer_to_tetranary_function(
            _Result (*__x)(_Arg1, _Arg2, _Arg3, _Arg4)) : _M_ptr(__x) {}
        _Result operator()(_Arg1 __x1, _Arg2 __x2, _Arg3 __x3, _Arg4 __x4) const 
        { return _M_ptr(__x1, __x2, __x3, __x4); }
    };

    template <class _Arg1, class _Arg2, class _Arg3, class _Arg4, class _Result>
    inline pointer_to_tetranary_function<_Arg1,_Arg2,_Arg3,_Arg4,_Result> ptr_fun(
        _Result (*__x)(_Arg1, _Arg2, _Arg3, _Arg4)) 
    {
        return pointer_to_tetranary_function<_Arg1,_Arg2,_Arg3,_Arg4,_Result>(__x);
    }

#if 0
    // These are the STL for Project in case I want to extend them to the other
    // function types some day
    // project1st and project2nd are extensions: they are not part of the standard
    template <class _Arg1, class _Arg2>
    struct _Project1st : public std::binary_function<_Arg1, _Arg2, _Arg1> 
    {
        _Arg1 operator()(const _Arg1& __x, const _Arg2&) const { return __x; }
    };

    template <class _Arg1, class _Arg2>
    struct _Project2nd : public std::binary_function<_Arg1, _Arg2, _Arg2> 
    {
        _Arg2 operator()(const _Arg1&, const _Arg2& __y) const { return __y; }
    };

    template <class _Arg1, class _Arg2> 
    struct project1st : public _Project1st<_Arg1, _Arg2> {};

    template <class _Arg1, class _Arg2>
    struct project2nd : public _Project2nd<_Arg1, _Arg2> {};
#endif

    // constant_void_fun, constant_unary_fun, and constant_binary_fun are
    // extensions: they are not part of the standard.  (The same, of course,
    // is true of the helper functions constant0, constant1, and constant2.)

#ifndef GCCCOMP
    template <class _Result>
    struct constant_nullary_fun : public nullary_function<_Result>
    {
        _Result __val;
        constant_nullary_fun(const _Result& __v) : __val(__v) {}
        const _Result& operator()() const { return __val; }
    };  

#ifndef __STL_LIMITED_DEFAULT_TEMPLATES
    template <class _Result, class _Argument = _Result>
    #else
    template <class _Result, class _Argument>
    #endif
    struct constant_unary_fun : public std::unary_function<_Argument, _Result> 
    {
        _Result _M_val;
        constant_unary_fun(const _Result& __v) : _M_val(__v) {}
        const _Result& operator()(const _Argument&) const { return _M_val; }
    };

#ifndef __STL_LIMITED_DEFAULT_TEMPLATES
    template <class _Result, class _Arg1 = _Result, class _Arg2 = _Arg1>
    #else
    template <class _Result, class _Arg1, class _Arg2>
    #endif
    struct constant_binary_fun : public std::binary_function<_Arg1, _Arg2, _Result> {
        _Result _M_val;
        constant_binary_fun(const _Result& __v) : _M_val(__v) {}
        const _Result& operator()(const _Arg1&, const _Arg2&) const 
        { return _M_val; }
    };

    template <class _Result>
    inline constant_nullary_fun<_Result> constant0(const _Result& __val)
    {
        return constant_nullary_fun<_Result>(__val);
    }

    template <class _Result>
    inline constant_unary_fun<_Result,_Result> constant1(const _Result& __val)
    {
        return constant_unary_fun<_Result,_Result>(__val);
    }

    template <class _Result>
    inline constant_binary_fun<_Result,_Result,_Result> constant2(
        const _Result& __val)
    {
        return constant_binary_fun<_Result,_Result,_Result>(__val);
    }
#endif


#if 0
    // These I might need someday, but for now I don't feel like
    // extending these.  So, again, here is the STL for mem_fun*
    // Adaptor function objects: pointers to member functions.

    // There are a total of 16 = 2^4 function objects in this family.
    //  (1) Member functions taking no arguments vs member functions taking
    //       one argument.
    //  (2) Call through pointer vs call through reference.
    //  (3) Member function with void return type vs member function with
    //      non-void return type.
    //  (4) Const vs non-const member function.

    // Note that choice (3) is nothing more than a workaround: according
    //  to the draft, compilers should handle void and non-void the same way.
    //  This feature is not yet widely implemented, though.  You can only use
    //  member functions returning void if your compiler supports partial
    //  specialization.

    // All of this complexity is in the function objects themselves.  You can
    //  ignore it by using the helper function mem_fun and mem_fun_ref,
    //  which create whichever type of adaptor is appropriate.
    //  (mem_fun1 and mem_fun1_ref are no longer part of the C++ standard,
    //  but they are provided for backward compatibility.)


    template <class _Ret, class _Tp>
    class mem_fun_t : public std::unary_function<_Tp*,_Ret> 
    {
    public:
        explicit mem_fun_t(_Ret (_Tp::*__pf)()) : _M_f(__pf) {}
        _Ret operator()(_Tp* __p) const { return (__p->*_M_f)(); }
    private:
        _Ret (_Tp::*_M_f)();
    };

    template <class _Ret, class _Tp>
    class const_mem_fun_t : public std::unary_function<const _Tp*,_Ret> 
    {
    public:
        explicit const_mem_fun_t(_Ret (_Tp::*__pf)() const) : _M_f(__pf) {}
        _Ret operator()(const _Tp* __p) const { return (__p->*_M_f)(); }
    private:
        _Ret (_Tp::*_M_f)() const;
    };


    template <class _Ret, class _Tp>
    class mem_fun_ref_t : public std::unary_function<_Tp,_Ret> 
    {
    public:
        explicit mem_fun_ref_t(_Ret (_Tp::*__pf)()) : _M_f(__pf) {}
        _Ret operator()(_Tp& __r) const { return (__r.*_M_f)(); }
    private:
        _Ret (_Tp::*_M_f)();
    };

    template <class _Ret, class _Tp>
    class const_mem_fun_ref_t : public std::unary_function<_Tp,_Ret> 
    {
    public:
        explicit const_mem_fun_ref_t(_Ret (_Tp::*__pf)() const) : _M_f(__pf) {}
        _Ret operator()(const _Tp& __r) const { return (__r.*_M_f)(); }
    private:
        _Ret (_Tp::*_M_f)() const;
    };

    template <class _Ret, class _Tp, class _Arg>
    class mem_fun1_t : public std::binary_function<_Tp*,_Arg,_Ret> 
    {
    public:
        explicit mem_fun1_t(_Ret (_Tp::*__pf)(_Arg)) : _M_f(__pf) {}
        _Ret operator()(_Tp* __p, _Arg __x) const { return (__p->*_M_f)(__x); }
    private:
        _Ret (_Tp::*_M_f)(_Arg);
    };

    template <class _Ret, class _Tp, class _Arg>
    class const_mem_fun1_t : public std::binary_function<const _Tp*,_Arg,_Ret> 
    {
    public:
        explicit const_mem_fun1_t(_Ret (_Tp::*__pf)(_Arg) const) : _M_f(__pf) {}
        _Ret operator()(const _Tp* __p, _Arg __x) const
        { return (__p->*_M_f)(__x); }
    private:
        _Ret (_Tp::*_M_f)(_Arg) const;
    };

    template <class _Ret, class _Tp, class _Arg>
    class mem_fun1_ref_t : public std::binary_function<_Tp,_Arg,_Ret> 
    {
    public:
        explicit mem_fun1_ref_t(_Ret (_Tp::*__pf)(_Arg)) : _M_f(__pf) {}
        _Ret operator()(_Tp& __r, _Arg __x) const { return (__r.*_M_f)(__x); }
    private:
        _Ret (_Tp::*_M_f)(_Arg);
    };

    template <class _Ret, class _Tp, class _Arg>
    class const_mem_fun1_ref_t : public std::binary_function<_Tp,_Arg,_Ret> 
    {
    public:
        explicit const_mem_fun1_ref_t(_Ret (_Tp::*__pf)(_Arg) const) : _M_f(__pf) {}
        _Ret operator()(const _Tp& __r, _Arg __x) const { return (__r.*_M_f)(__x); }
    private:
        _Ret (_Tp::*_M_f)(_Arg) const;
    };

#ifdef __STL_CLASS_PARTIAL_SPECIALIZATION

    template <class _Tp>
    class mem_fun_t<void, _Tp> : public std::unary_function<_Tp*,void> 
    {
    public:
        explicit mem_fun_t(void (_Tp::*__pf)()) : _M_f(__pf) {}
        void operator()(_Tp* __p) const { (__p->*_M_f)(); }
    private:
        void (_Tp::*_M_f)();
    };

    template <class _Tp>
    class const_mem_fun_t<void, _Tp> : public std::unary_function<const _Tp*,void> 
    {
    public:
        explicit const_mem_fun_t(void (_Tp::*__pf)() const) : _M_f(__pf) {}
        void operator()(const _Tp* __p) const { (__p->*_M_f)(); }
    private:
        void (_Tp::*_M_f)() const;
    };

    template <class _Tp>
    class mem_fun_ref_t<void, _Tp> : public std::unary_function<_Tp,void> 
    {
    public:
        explicit mem_fun_ref_t(void (_Tp::*__pf)()) : _M_f(__pf) {}
        void operator()(_Tp& __r) const { (__r.*_M_f)(); }
    private:
        void (_Tp::*_M_f)();
    };

    template <class _Tp>
    class const_mem_fun_ref_t<void, _Tp> : public std::unary_function<_Tp,void> 
    {
    public:
        explicit const_mem_fun_ref_t(void (_Tp::*__pf)() const) : _M_f(__pf) {}
        void operator()(const _Tp& __r) const { (__r.*_M_f)(); }
    private:
        void (_Tp::*_M_f)() const;
    };

    template <class _Tp, class _Arg>
    class mem_fun1_t<void, _Tp, _Arg> : public std::binary_function<_Tp*,_Arg,void> 
    {
    public:
        explicit mem_fun1_t(void (_Tp::*__pf)(_Arg)) : _M_f(__pf) {}
        void operator()(_Tp* __p, _Arg __x) const { (__p->*_M_f)(__x); }
    private:
        void (_Tp::*_M_f)(_Arg);
    };

    template <class _Tp, class _Arg>
    class const_mem_fun1_t<void, _Tp, _Arg> : 
        public std::binary_function<const _Tp*,_Arg,void> 
    {
    public:
        explicit const_mem_fun1_t(void (_Tp::*__pf)(_Arg) const) : _M_f(__pf) {}
        void operator()(const _Tp* __p, _Arg __x) const { (__p->*_M_f)(__x); }
    private:
        void (_Tp::*_M_f)(_Arg) const;
    };

    template <class _Tp, class _Arg>
    class mem_fun1_ref_t<void, _Tp, _Arg> : 
        public std::binary_function<_Tp,_Arg,void> 
    {
    public:
        explicit mem_fun1_ref_t(void (_Tp::*__pf)(_Arg)) : _M_f(__pf) {}
        void operator()(_Tp& __r, _Arg __x) const { (__r.*_M_f)(__x); }
    private:
        void (_Tp::*_M_f)(_Arg);
    };

    template <class _Tp, class _Arg>
    class const_mem_fun1_ref_t<void, _Tp, _Arg> :
        public std::binary_function<_Tp,_Arg,void> 
    {
    public:
        explicit const_mem_fun1_ref_t(void (_Tp::*__pf)(_Arg) const) : _M_f(__pf) {}
        void operator()(const _Tp& __r, _Arg __x) const { (__r.*_M_f)(__x); }
    private:
        void (_Tp::*_M_f)(_Arg) const;
    };

#endif // __STL_CLASS_PARTIAL_SPECIALIZATION 

    // Mem_fun adaptor helper functions.  There are only two:
    //  mem_fun and mem_fun_ref.  (mem_fun1 and mem_fun1_ref 
    //  are provided for backward compatibility, but they are no longer
    //  part of the C++ standard.)

    template <class _Ret, class _Tp>
    inline mem_fun_t<_Ret,_Tp> mem_fun(_Ret (_Tp::*__f)())
    { return mem_fun_t<_Ret,_Tp>(__f); }

    template <class _Ret, class _Tp>
    inline const_mem_fun_t<_Ret,_Tp> mem_fun(_Ret (_Tp::*__f)() const)
    { return const_mem_fun_t<_Ret,_Tp>(__f); }

    template <class _Ret, class _Tp>
    inline mem_fun_ref_t<_Ret,_Tp> mem_fun_ref(_Ret (_Tp::*__f)()) 
    { return mem_fun_ref_t<_Ret,_Tp>(__f); }

    template <class _Ret, class _Tp>
    inline const_mem_fun_ref_t<_Ret,_Tp> mem_fun_ref(_Ret (_Tp::*__f)() const)
    { return const_mem_fun_ref_t<_Ret,_Tp>(__f); }

    template <class _Ret, class _Tp, class _Arg>
    inline mem_fun1_t<_Ret,_Tp,_Arg> mem_fun(_Ret (_Tp::*__f)(_Arg))
    { return mem_fun1_t<_Ret,_Tp,_Arg>(__f); }

    template <class _Ret, class _Tp, class _Arg>
    inline const_mem_fun1_t<_Ret,_Tp,_Arg> mem_fun(_Ret (_Tp::*__f)(_Arg) const)
    { return const_mem_fun1_t<_Ret,_Tp,_Arg>(__f); }

    template <class _Ret, class _Tp, class _Arg>
    inline mem_fun1_ref_t<_Ret,_Tp,_Arg> mem_fun_ref(_Ret (_Tp::*__f)(_Arg))
    { return mem_fun1_ref_t<_Ret,_Tp,_Arg>(__f); }

    template <class _Ret, class _Tp, class _Arg>
    inline const_mem_fun1_ref_t<_Ret,_Tp,_Arg>
    mem_fun_ref(_Ret (_Tp::*__f)(_Arg) const)
    { return const_mem_fun1_ref_t<_Ret,_Tp,_Arg>(__f); }

    template <class _Ret, class _Tp, class _Arg>
    inline mem_fun1_t<_Ret,_Tp,_Arg> mem_fun1(_Ret (_Tp::*__f)(_Arg))
    { return mem_fun1_t<_Ret,_Tp,_Arg>(__f); }

    template <class _Ret, class _Tp, class _Arg>
    inline const_mem_fun1_t<_Ret,_Tp,_Arg> mem_fun1(_Ret (_Tp::*__f)(_Arg) const)
    { return const_mem_fun1_t<_Ret,_Tp,_Arg>(__f); }

    template <class _Ret, class _Tp, class _Arg>
    inline mem_fun1_ref_t<_Ret,_Tp,_Arg> mem_fun1_ref(_Ret (_Tp::*__f)(_Arg))
    { return mem_fun1_ref_t<_Ret,_Tp,_Arg>(__f); }

    template <class _Ret, class _Tp, class _Arg>
    inline const_mem_fun1_ref_t<_Ret,_Tp,_Arg>
    mem_fun1_ref(_Ret (_Tp::*__f)(_Arg) const)
    { return const_mem_fun1_ref_t<_Ret,_Tp,_Arg>(__f); }
#endif


    //
    // Define compose functions.  See Josuttis, pp. 313-320.
    // compose_f_gx     is f(g(x))      aka compose1
    // compose_f_gxy    is f(g(x,y))
    // compose_f_gx_hx  is f(g(x),h(x)) aka compose2
    // compose_f_gx_gy  is f(g(x),g(y)) 
    // compose_f_gx_hy  is f(g(x),h(y))
    // compose_f_gx_y   is f(g(x),y)
    // compose_f_x_gy   is f(x,g(y))
    // Add others as needed
    //

    template <class OP1, class OP2>
    class compose_f_gx_t : 
        public std::unary_function<typename OP2::argument_type,
        typename OP1::result_type>
    {
    private:
        OP1 op1;
        OP2 op2;
    public:
        compose_f_gx_t(const OP1& o1, const OP2& o2) : op1(o1),op2(o2) {}
        typename OP1::result_type operator()(
            const typename OP2::argument_type& x) const 
        { return op1(op2(x)); }
    };

    template <class OP1,class OP2>
    inline compose_f_gx_t<OP1,OP2> compose_f_gx(const OP1& o1, const OP2& o2) 
    {
        return compose_f_gx_t<OP1,OP2>(o1,o2);
    }

    template <class OP1, class OP2, class OP3>
    class compose_f_gx_hx_t : 
        public std::unary_function<typename OP2::argument_type,
        typename OP1::result_type>
    {
    private:
        OP1 op1;
        OP2 op2;
        OP3 op3;
    public:
        compose_f_gx_hx_t(const OP1& o1, const OP2& o2, const OP3& o3) : 
            op1(o1),op2(o2),op3(o3) {}
        typename OP1::result_type operator()(
            const typename OP2::argument_type& x) const 
        { return op1(op2(x),op3(x)); }
    };

    template <class OP1,class OP2,class OP3>
    inline compose_f_gx_hx_t<OP1,OP2,OP3> compose_f_gx_hx(
        const OP1& o1, const OP2& o2, const OP3& o3) 
    {
        return compose_f_gx_hx_t<OP1,OP2,OP3>(o1,o2,o3);
    }

    template <class OP1, class OP2, class OP3>
    class compose_f_gx_hy_t :
        public std::binary_function<typename OP2::argument_type,
        typename OP3::argument_type,
        typename OP1::result_type>
    {
    private:
        OP1 op1;
        OP2 op2;
        OP3 op3;
    public:
        compose_f_gx_hy_t(const OP1& o1, const OP2& o2, const OP3& o3) : 
            op1(o1),op2(o2),op3(o3) {}
        typename OP1::result_type operator()(
            const typename OP2::argument_type& x,
            const typename OP3::argument_type& y) const 
        { return op1(op2(x),op3(y)); }
    };

    template <class OP1,class OP2,class OP3>
    inline compose_f_gx_hy_t<OP1,OP2,OP3> compose_f_gx_hy(
        const OP1& o1, const OP2& o2, const OP3& o3) 
    {
        return compose_f_gx_hy_t<OP1,OP2,OP3>(o1,o2,o3);
    }

    template <class OP1, class OP2>
    class compose_f_gx_gy_t :
        public std::binary_function<typename OP2::argument_type,
        typename OP2::argument_type,
        typename OP1::result_type>
    {
    private:
        OP1 op1;
        OP2 op2;
    public:
        compose_f_gx_gy_t(const OP1& o1, const OP2& o2) : op1(o1),op2(o2) {}
        typename OP1::result_type operator()(
            const typename OP2::argument_type& x,
            const typename OP2::argument_type& y) const 
        { return op1(op2(x),op2(y)); }
    };

    template <class OP1,class OP2>
    inline compose_f_gx_gy_t<OP1,OP2> compose_f_gx_gy(
        const OP1& o1, const OP2& o2) 
    {
        return compose_f_gx_gy_t<OP1,OP2>(o1,o2);
    }

    template <class OP1, class OP2>
    class compose_f_gxy_t : 
        public std::binary_function<typename OP2::first_argument_type,
        typename OP2::second_argument_type,
        typename OP1::result_type>
    {
    private:
        OP1 op1;
        OP2 op2;
    public:
        compose_f_gxy_t(const OP1& o1, const OP2& o2) : op1(o1),op2(o2) {}
        typename OP1::result_type operator()(
            const typename OP2::first_argument_type& x,
            const typename OP2::second_argument_type& y) const 
        { return op1(op2(x,y)); }
    };

    template <class OP1,class OP2>
    inline compose_f_gxy_t<OP1,OP2> compose_f_gxy(const OP1& o1, const OP2& o2) 
    {
        return compose_f_gxy_t<OP1,OP2>(o1,o2);
    }

    template <class OP1, class OP2, class YTYPE>
    class compose_f_gx_y_t : 
        public std::binary_function<typename OP2::argument_type,
        YTYPE,
        typename OP1::result_type>
    {
    private:
        OP1 op1;
        OP2 op2;
    public:
        compose_f_gx_y_t(const OP1& o1, const OP2& o2, const YTYPE& y) : 
            op1(o1),op2(o2) {}
        typename OP1::result_type operator()(
            const typename OP2::argument_type& x, const YTYPE& y) const 
        { return op1(op2(x),y); }
    };

    template <class OP1,class OP2,class YTYPE>
    inline compose_f_gx_y_t<OP1,OP2,YTYPE> compose_f_gx_y(
        const OP1& o1, const OP2& o2, const YTYPE& y) 
    {
        return compose_f_gx_y_t<OP1,OP2,YTYPE>(o1,o2,y);
    }

    template <class OP1, class OP2, class XTYPE>
    class compose_f_x_gy_t : 
        public std::binary_function<XTYPE,
        typename OP2::argument_type,
        typename OP1::result_type>
    {
    private:
        OP1 op1;
        OP2 op2;
    public:
        compose_f_x_gy_t(const OP1& o1, const OP2& o2, const XTYPE& x) : 
            op1(o1),op2(o2) {}
        typename OP1::result_type operator()(
            const XTYPE& x, const typename OP2::argument_type& y) const 
        { return op1(x,op2(y)); }
    };

    template <class OP1,class OP2,class XTYPE>
    inline compose_f_x_gy_t<OP1,OP2,XTYPE> compose_f_x_gy(
        const OP1& o1, const XTYPE& x, const OP2& o2) 
    {
        return compose_f_x_gy_t<OP1,OP2,XTYPE>(o1,o2,x);
    }

}}

#endif
