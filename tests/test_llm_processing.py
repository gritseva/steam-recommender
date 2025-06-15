# tests/test_llm_processing.py
import unittest
from unittest.mock import patch
import json

# Import the function we want to test
from utils.llm_processing import infer_user_preferences_with_llm


"""
Explanation
Happy Path Test:

We patch both the tokenizer and the model in utils.llm_processing so that when infer_user_preferences_with_llm is called, it doesn't execute the real transformer inference.
We configure model.generate to return dummy tokens and tokenizer.decode to return a well-formed JSON string.
We also simulate the output of tokenizer(...) to be a dummy dictionary.
Finally, we assert that the function returns the expected dictionary.
Edge Case Test (Invalid JSON):

We simulate a situation where tokenizer.decode returns an invalid JSON string.
We also patch extract_game_titles to return a predetermined value (e.g., ["Game B"]).
When JSON decoding fails, the function should fall back on extract_game_titles and default to empty lists for other keys.
The test asserts that the fallback behavior produces the expected result.
This approach demonstrates how to mock external dependencies such as the transformer model, tokenizer, and even file I/O or API calls (by using similar techniques) so your tests remain fast, deterministic, and isolated from external services.

You can run these tests with:

bash
python -m unittest discover -s tests
"""


class TestLLMProcessing(unittest.TestCase):
    @patch('utils.llm_processing.tokenizer')
    @patch('utils.llm_processing.model')
    def test_infer_user_preferences_with_llm_happy_path(self, mock_model, mock_tokenizer):
        """
        Test the happy path where the transformer model returns valid JSON.
        """
        # Define the expected output as a JSON string
        expected_preferences = {
            "liked_games": ["Game A"],
            "genres": ["Action"],
            "excluded_tags": []
        }
        expected_output = json.dumps(expected_preferences)

        # Set up the mocks:
        # When model.generate is called, return a dummy token sequence.
        fake_generated_tokens = [b"dummy"]
        mock_model.generate.return_value = fake_generated_tokens

        # When tokenizer.decode is called on the generated tokens, return the expected JSON string.
        mock_tokenizer.decode.return_value = expected_output

        # Also, simulate the tokenizer() call that prepares the inputs (returns dummy tensor-like dict).
        mock_tokenizer.return_value = {'input_ids': [
            1, 2, 3], 'attention_mask': [1, 1, 1]}

        # Now, call the function with a sample user message.
        user_message = "I really love Game A! It's fantastic."
        preferences = infer_user_preferences_with_llm(user_message)

        # Assert that the function returns the expected preferences dictionary.
        self.assertEqual(preferences, expected_preferences)

    @patch('utils.llm_processing.extract_game_titles', return_value=["Game B"])
    @patch('utils.llm_processing.tokenizer')
    @patch('utils.llm_processing.model')
    def test_infer_user_preferences_with_llm_invalid_json(self, mock_model, mock_tokenizer, mock_extract):
        """
        Test the edge case where the transformer model returns an invalid JSON string.
        In that case, the function should fallback to calling extract_game_titles.
        """
        # Simulate model.generate returning dummy tokens.
        fake_generated_tokens = [b"dummy"]
        mock_model.generate.return_value = fake_generated_tokens

        # Simulate tokenizer.decode returning an invalid JSON string.
        invalid_output = "invalid json"
        mock_tokenizer.decode.return_value = invalid_output

        # Again, simulate the tokenizer() call for input preparation.
        mock_tokenizer.return_value = {'input_ids': [
            1, 2, 3], 'attention_mask': [1, 1, 1]}

        # Call the function with a sample message.
        user_message = "Some message that leads to an invalid JSON output."
        preferences = infer_user_preferences_with_llm(user_message)

        # Since JSON decoding fails, the function should fall back on extract_game_titles,
        # which we patched to return ["Game B"]. Other keys should default to empty lists.
        expected_preferences = {"liked_games": [
            "Game B"], "genres": [], "excluded_tags": []}
        self.assertEqual(preferences, expected_preferences)


if __name__ == '__main__':
    unittest.main()
