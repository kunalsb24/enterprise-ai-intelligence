from unittest.mock import MagicMock, patch

import torch

from enterprise_ai.llm.model import LocalLLM


@patch("enterprise_ai.llm.model.AutoModelForCausalLM")
@patch("enterprise_ai.llm.model.AutoTokenizer")
def test_local_llm_generates_response(
    mock_tokenizer_class,
    mock_model_class,
):
    tokenizer = MagicMock()
    model = MagicMock()

    mock_tokenizer_class.from_pretrained.return_value = tokenizer
    mock_model_class.from_pretrained.return_value = model

    tokenizer.apply_chat_template.return_value = "formatted prompt"

    tokenizer.return_value = {
        "input_ids": torch.tensor([[10, 20, 30]])
    }

    model.generate.return_value = torch.tensor(
        [[10, 20, 30, 40, 50]]
    )

    tokenizer.decode.return_value = "Generated answer"

    llm = LocalLLM(model_name="test-model")

    result = llm.generate(
        messages=[
            {
                "role": "user",
                "content": "Test question",
            }
        ],
        max_new_tokens=50,
    )

    assert result == "Generated answer"

    mock_tokenizer_class.from_pretrained.assert_called_once_with(
        "test-model"
    )

    mock_model_class.from_pretrained.assert_called_once_with(
        "test-model",
        dtype=torch.float32,
    )

    model.generate.assert_called_once()

    tokenizer.decode.assert_called_once()

    decode_args, decode_kwargs = tokenizer.decode.call_args

    assert torch.equal(
        decode_args[0],
        torch.tensor([40, 50]),
    )

    assert decode_kwargs == {
        "skip_special_tokens": True,
    }