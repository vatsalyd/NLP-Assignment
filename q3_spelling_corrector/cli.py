import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from q3_spelling_corrector.spelling_corrector import (
    load_brown_corpus, build_models, edit_distance_1, 
    SymmetricDeleteCorrector, correct_nonword, correct_realword
)
import time

def main():
    print("Loading models...")
    sentences, words = load_brown_corpus()
    vocab, unigram_counts, unigram_probs, bigram_probs = build_models(words)
    sym_delete = SymmetricDeleteCorrector(vocab, unigram_counts)
    print("Ready! Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("Enter sentence: ").strip()
            if user_input.lower() == 'exit':
                break
            
            if not user_input:
                continue
            
            start = time.perf_counter()
            
            words_input = user_input.split()
            corrected_words = []
            changes = []
            
            for i, word in enumerate(words_input):
                clean_word = word.lower().strip('.,!?;:')
                punct = word[len(clean_word):] if len(word) > len(clean_word) else ''
                
                if clean_word not in vocab:
                    corrected = correct_nonword(clean_word, vocab, unigram_counts, edit_distance_1, sym_delete)
                else:
                    prev = words_input[i-1].lower().strip('.,!?;:') if i > 0 else ''
                    corrected = correct_realword(clean_word, prev, vocab, unigram_counts, bigram_probs, edit_distance_1, sym_delete)
                
                corrected_word = corrected + punct
                corrected_words.append(corrected_word)
                
                if corrected != clean_word:
                    changes.append((word, corrected_word))
            
            elapsed = (time.perf_counter() - start) * 1000
            
            print(f"Corrected: {' '.join(corrected_words)}")
            if changes:
                print("Changes:", ', '.join(f"**{o}** -> **{c}**" for o, c in changes))
            print(f"Latency: {elapsed:.2f}ms\n")
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print("Goodbye!")

if __name__ == "__main__":
    main()