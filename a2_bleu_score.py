'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall

All of the files in this directory and all subdirectories are:
Copyright (c) 2021 University of Toronto
'''

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x


def grouper(seq, n):
    '''Get all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of token ids or words representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    last_index = len(seq) - n
    ngrams = []
    for i in range(0,last_index+1):
        ngrams = ngrams + [seq[i:i+n]]
    return ngrams


def n_gram_precision(reference, candidate, n):
    '''Compute the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''
    if len(candidate) < n:
        return 0
    reference_gms = grouper(reference, n)
    candidate_gms = grouper(candidate, n)

    count = 0
    for gram in candidate_gms:
        if gram in reference_gms:
            count = count + 1
    return float(count)/float(len(candidate_gms))



def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    if len(candidate) == 0:
        return 0
    metric = float(len(reference))/len(candidate)
    if metric < 1:
        return 1
    else:
        return float(exp(1-metric))


def BLEU_score(reference, candidate, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    mul = 1.0
    for i in range(1, n+1):
        mul = mul * n_gram_precision(reference, candidate, i)
    
    return float(brevity_penalty(reference, candidate)*(mul**(1/n)))
