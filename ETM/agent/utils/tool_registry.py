"""
Tool Registry

Manages tools available to the Agent, including registration, invocation, and description.
"""

from typing import Dict, Any, List, Callable, Optional


class ToolRegistry:
    """
    Tool registry that manages tools available to the Agent.
    """
    
    def __init__(self):
        """
        Initialize tool registry.
        """
        self.tools = {}
        self.descriptions = {}
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str
    ) -> None:
        """
        Register a tool.
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
        """
        self.tools[name] = func
        self.descriptions[name] = description
    
    def call_tool(
        self,
        name: str,
        args: Dict[str, Any]
    ) -> Any:
        """
        Call a tool.
        
        Args:
            name: Tool name
            args: Tool arguments
            
        Returns:
            Tool call result
        """
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        
        return self.tools[name](**args)
    
    def get_tool_description(
        self,
        name: str
    ) -> str:
        """
        Get tool description.
        
        Args:
            name: Tool name
            
        Returns:
            Tool description
        """
        if name not in self.descriptions:
            raise ValueError(f"Tool '{name}' not found")
        
        return self.descriptions[name]
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools.
        
        Returns:
            List of tools
        """
        return [
            {
                "name": name,
                "description": self.descriptions[name]
            }
            for name in self.tools
        ]
    
    def register_default_tools(self) -> None:
        """
        Register default tools.
        """
        # Search tool
        self.register_tool(
            name="search",
            func=self._search_tool,
            description="Search for relevant information"
        )
        
        # Calculate tool
        self.register_tool(
            name="calculate",
            func=self._calculate_tool,
            description="Perform mathematical calculations"
        )
        
        # Date tool
        self.register_tool(
            name="get_date",
            func=self._date_tool,
            description="Get current date and time"
        )
    
    def _search_tool(self, query: str) -> Dict[str, Any]:
        """
        Search tool implementation.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        # This is a simple mock implementation
        # In actual application, can integrate with search API
        return {
            "query": query,
            "results": [
                {"title": f"Result for {query} 1", "snippet": f"This is a snippet about {query}..."},
                {"title": f"Result for {query} 2", "snippet": f"More information about {query}..."}
            ]
        }
    
    def _calculate_tool(self, expression: str) -> Dict[str, Any]:
        """
        Calculate tool implementation.
        
        Args:
            expression: Mathematical expression
            
        Returns:
            Calculation result
        """
        try:
            # Safe eval implementation
            import ast
            import operator
            
            # Supported operators
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.BitXor: operator.xor,
                ast.USub: operator.neg
            }
            
            def eval_expr(expr):
                return eval_(ast.parse(expr, mode='eval').body)
            
            def eval_(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return operators[type(node.op)](eval_(node.left), eval_(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return operators[type(node.op)](eval_(node.operand))
                else:
                    raise TypeError(node)
            
            result = eval_expr(expression)
            
            return {
                "expression": expression,
                "result": result,
                "success": True
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "success": False
            }
    
    def _date_tool(self) -> Dict[str, Any]:
        """
        Date tool implementation.
        
        Returns:
            Current date and time
        """
        import datetime
        
        now = datetime.datetime.now()
        
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "timestamp": now.timestamp()
        }
