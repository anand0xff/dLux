import equinox as eqx
import typing


PyTree = typing.NewType("PyTree", object)
PyTreeLeaf = typing.NewType("PyTreeLeaf", object)

NotALeaf = typing.NewType("NotALeaf", Exception)

GradientInterface = typing.NewType("GradientInterface", object)


# TODO: Talk to @LouisDesdoigts about the custom exceptions and 
# see if he thinks that it is overkill.
class NotALeaf(Exception):
    """
    To be raised when someone attempts to access a leaf that does 
    not exist. 

    Attributes
    ----------
    pytree : PyTree
        The `PyTree` that someone tried to access at a non-extant 
        leaf.
    leaf : PyTreeLeaf
        The `PyTreeLeaf` that does not exist which someone tried 
        to access.
    """
    pytree : PyTree
    leaf : PyTreeLeaf


    def __init__(self : NotALeaf, pytree : PyTree, 
            leaf : PyTreeLeaf) -> NotALeaf:
        """
        Parameters
        ----------
        message : str
            A helpful error message to be printed to the console. 
        pytree : PyTree
            The offending `PyTree` that caused the exception.
        leaf : PyTreeLeaf
            The `PyTreeLeaf` that was accessed without existing.

        Returns
        -------
        : NotALeaf
            A helpful debugging exception.
        """
        self.pytree = pytree
        self.leaf = leaf


    def __str__(self : NotALeaf) -> str:
        """
        Formats the `PyTree` showing the leaves that do exist as
        well as the leaf the does not exist.

        Returns
        -------
        : str
            The helpfully formatted error message. 
        """
        print_string = \
            f"""{self.leaf} is not a leaf of {self.pytree}"""


class GradientInterface(eqx.Module):
    """
    Implements useful functionality for working with gradients 
    through `equinox.Modules`. This class is abstract and should 
    never be instantiated directly.
    """
    def _clone_pytree_as_false(self : GradientInterface) -> PyTree:
        """
        Convinience function for generating a `PyTree` with all
        false leaves that has the same structure as `self`. This 
        is designed to be used with `self._gradient_leaf()` and 
        is not part of the public interface.

        Returns
        -------
        : PyTree
            A `PyTree` with a matching structure to `self` and all
            leaves set to `false`.
        """
        # TODO: Confirm that this actually works
        return eqx.tree_map(lambda : False, self)


    def _set_leaf_as_true(self : GradientInterface, 
            parameter : str, filtered_leaves : PyTree) -> PyTree:
        """
        Allows `grad`s and `jit`s to be taken with respect to 
        `parameter`. 

        Parameters
        ----------
        parameter : str
            The parameter that you want to take the gradient with
            respect to. 
        fitered_leaves : PyTree
            A `PyTree` describing the parameters that you want to 
            be differentiable. Updated by this transformation.

        Returns
        -------
        : PyTree
            A `PyTree` of the same structure as `self` with the 
            leaf corresponding to parameter set as `True`. The 
            other leaves may be set as `True` or `False` but 
            must be `bool` data types.
        """
        # TODO: Discuss error handling with @LouisDesdoigts 
        # Create our own very friendly dLux errors.
        if not filetered_leaves.__dict__.contains(parameter): 
            raise NotALeaf(filtered_leaves, parameter)

        return eqx.tree_at(
            lambda pytree : pytree.__dict__[parameter], 
            filtered_leaves, replace = True)


    def gradients_wrt(self : GradientInterface, 
            parameters : list) -> PyTree:
        """
        Create a filter_spec/arg argument to be used with equinox 
        that has the correct structure to define the gradients.

        Parameters
        ----------
        parameters : list[str]
            A list of parameter names as they appear in `self` 
            that you wish to take gradients with respect to.

        Returns
        -------
        : PyTree
            A boolean pytree with the appropriated leaves set 
            to True and all others False.
        """
        pytree = self._clone_pytree_as_false()
        # functools.reduce would be perfect but it does not cope 
        # with the carry.
        # jax.lax.scan may be a possiblity .Not a possiblity.
        for parameter in parameters:
            pytree = self._set_leaf_as_true(parameter)
        
        return pytree  
