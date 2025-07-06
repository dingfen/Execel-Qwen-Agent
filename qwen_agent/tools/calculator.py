from qwen_agent.tools.base import BaseTool, register_tool
import json5

@register_tool('calculator')
class Calculator(BaseTool):
    """A calculator tool."""

    name: str = 'calculator'
    description: str = 'A simple calculator tool.'
    parameters = [{
        'name': 'formula',
        'type': 'string'
    }]

    def call(self, params : str) -> float:
        return eval(json5.load(params)['formula'])
    