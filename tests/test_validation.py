import pytest

from rxnutils.routes.utils.validation import validate_dict


@pytest.mark.parametrize(
    "route_index",
    [0, 1, 2],
)
def test_validate_example_trees(load_reaction_tree, route_index):
    validate_dict(load_reaction_tree("example_routes.json", route_index))


def test_validate_only_mols():
    dict_ = {
        "smiles": "CCC",
        "type": "mol",
        "children": [{"smiles": "CCC", "type": "mol"}],
    }
    with pytest.raises(ValueError, match="String should match pattern"):
        validate_dict(dict_)
