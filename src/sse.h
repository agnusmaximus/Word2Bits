// Copyright (C) 2009-2013 Mischa Sandberg <mischasan@gmail.com>
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License Version 2 as
// published by the Free Software Foundation.  You may not use, modify or
// distribute this program under any other version of the GNU General
// Public License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
//
// IF YOU ARE UNABLE TO WORK WITH GPL2, CONTACT ME.
//-------------------------------------------------------------------
#ifndef SSE_H
#define SSE_H

#include <emmintrin.h>  // _mm_set_epi64x

#if defined(__linux__)
#   define regargs __attribute__((fastcall))
#else
#   define regargs
#endif

//--------------|---------------------------------------------
// xm_* functions/macros are abbreviations for common ops,
//  avoiding sometimes-cryptic _mm_* intrinsic names.
//
// xm_diff(x,y)         - 16bit mask with bits set where bytes of two xmm values differ.
// xm_same(x,y)         - 16bit mask with bits set where bytes of two xmm values match.
// xm_fill(b)           - Repeat (byte) x 16 as an xmm value.
// xm_load(XMM const*)  - LOad/AligneD xmm value.
// xm_loud(void const*) - LOad/UnaligneD xmm value.
// xm_zero()            - (XMM) 0x000...000
// xm_ones()            - (XMM) 0xFFF...FFF
// xm_{and,andnot,not,or,xor} - standard bit ops for XMM.
// xm_f[fl]s(x)         - find first/last bit set (0..127) in an XMM value.
//                          xm_f[fl]s(xm_zero()) = -1
// xm_sh[lr](x,n)          - 128-bit (left,right) variable shift
// xm_sh[lr]_001(x) ... xm_sh[lr]_177(x) - optimal inline constant (left,right) shifts.

// Format x as a string; return "buf"
// xm_dbl(x,buf) - two doubles (lo,hi)
// xm_hex(x,buf) - MSB-first 32-hexit
// xm_str(x,buf) - LSB-first hex "xx,xx,xx,..,xx-xx,xx,xx,..,xx"

typedef __m128i XMM;

#define xm_fill(c)    _mm_set1_epi8(c)

#define xm_and(a,b)    _mm_and_si128(a,b)
#define xm_andnot(a,b) _mm_andnot_si128(a,b)    // AKA "bic"
#define xm_not(a)      xm_xor(a, xm_ones)
#define xm_or(a,b)     _mm_or_si128(a,b)
#define xm_xor(a,b)    _mm_xor_si128(a,b)

static inline unsigned xm_same(XMM a, XMM b)
{ return _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)); }

static inline unsigned xm_diff(XMM a, XMM b)
{ return xm_same(a, b) ^ 0xFFFF; }

static inline XMM xm_load(void const*p)
{ return _mm_load_si128((XMM const*)p); }

static inline XMM xm_loud(void const*p)
{ return (XMM)_mm_loadu_pd((double const*)p); }

static inline XMM xm_zero(void)
{ return _mm_setzero_si128(); }

static inline XMM xm_ones(void)
{ XMM x = {}; return _mm_cmpeq_epi8(x,x); }

int xm_ffs(XMM x);
int xm_fls(XMM x);

XMM xm_shl(XMM x, unsigned nbits);
XMM xm_shr(XMM x, unsigned nbits);

char *xm_dbl(__m128d x, char buf[48]);
char *xm_hex(XMM x, char buf[48]);
char *xm_str(XMM x, char buf[48]);

// CPP tricks to generate static inline functions:
//      xm_shl_001 .. xm_shl_177 and xm_shr_001 .. xm_shr_177
// which can be called directly, and are also used in xm_shl() and xm_shr().
// ops: 15(1) 56(2) 7(5) 49(6) =

#define DO_7x7(A)    DO_7(A,1)    DO_7(A,2)    DO_7(A,3)    DO_7(A,4)    DO_7(A,5)    DO_7(A,6)    DO_7(A,7)
#define DO_7(A,B)    DO_LR(A,B,1) DO_LR(A,B,2) DO_LR(A,B,3) DO_LR(A,B,4) DO_LR(A,B,5) DO_LR(A,B,6) DO_LR(A,B,7)
#define DO_LR(A,B,C) DO(A,B,C,shl,shr) DO(A,B,C,shr,shl)

#define xm_bshl(x,n)  _mm_slli_si128(x,n) // xm <<= 8*n  -- BYTE shift
#define xm_bshr(x,n)  _mm_srli_si128(x,n) // xm <<= 8*n  -- BYTE shift
#define xm_shl64(x,n) _mm_slli_epi64(x,n) // xm.hi <<= n, xm.lo <<= n
#define xm_shr64(x,n) _mm_srli_epi64(x,n) // xm.hi >>= n, xm.lo >>= n

#undef  DO
#define DO(A,B,C,FWD,BAK) \
    static inline XMM xm_##FWD##_00##C(XMM x) \
    { return xm_or(xm_##FWD##64(x, 0##C), \
                   xm_##BAK##64(xm_b##FWD(x, 8), 64-0##C)); }
DO_7(0,0)       // 1..7

#undef  DO
#define DO(A,B,C,FWD,BAK) \
    static inline XMM xm_##FWD##_##A##C##0(XMM x) \
    { return xm_b##FWD(x, 0##A##C); }
DO_7(0,0)       // 8,16 .. 56
DO_LR(1,0,0)    // 64
DO_7(1,0)       // 72,80 .. 120

#undef  DO
#define DO(A,B,C,FWD,BAK) \
    static inline XMM xm_##FWD##_##A##B##C(XMM x) \
    { return xm_or(xm_##FWD##64(xm_b##FWD(x, 0##A##B), 0##C), \
                   xm_##BAK##64(xm_b##FWD(x, 0##A##B+8), 64-0##A##B)); }
DO_7x7(0)       // 9..63 except 16,24, ...

#undef  DO
#define DO(A,B,C,FWD,BAK) \
    static inline XMM xm_##FWD##_##A##B##C(XMM x) \
    { return xm_##FWD##64(xm_b##FWD(x, 0##A##B), 0##C); }
DO_7(1,0)       // 65..71
DO_7x7(1)       // 73..127 except 80,88,...

//--------------|---------------------------------------------
// These are more substantial uses of SSE2

// ssebmx: bit matrix transpose. nrows and ncols must be multiples of 8.
void    ssebmx(char const *inp, char *out, int nrows, int ncols);

// ssebmx_m: Same as ssebmx, but treat bytes as msbit-first.
void    ssebmx_m(char const *inp, char *out, int nrows, int ncols);

// ssebndm: memmem, using SSE2 for patlen in [65..128]
char   *ssebndm(char *target, int tgtlen, char *pattern, int patlen);

// ssecmp: strcmp implemented using SSE2
int     ssecmp(char const *s, char const *t);

// ssestr: strstr implemented using SSE2 and ssechr2
char const *ssestr(char const *tgt, char const *pat);

// ssechr2: used by ssestr; useful as a special case.
char const *ssechr2(char const *tgt, char const pat[2]) regargs;

// ssesort16d: sort 16 doubles in-place:
void    ssesort16d(double keys[16]);

// sserank16d: generate ranking vector for keys[],
//  such that for the input value of keys[]:
//      keys[rank[i]] <= keys[rank[j]]  IFF  i <= j
//  and the sort is stable; i.e.:
//      rank[i] < rank[j] if < j AND keys[rank[i]] == keys[rank[j]]
// "keys" is sorted as a side effect.
void    sserank16d(double keys[16], int rank[16]);

#endif //SSE_H
