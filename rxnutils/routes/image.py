""" This module contains a collection of routines to produce pretty images
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw

if TYPE_CHECKING:
    # pylint: disable=ungrouped-imports
    from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

    PilColor = Union[str, Tuple[int, int, int]]
    FrameColors = Optional[Dict[bool, PilColor]]
    from PIL.Image import Image as PilImage


def molecule_to_image(mol: Chem.rdchem.Mol, frame_color: PilColor, size: int = 300) -> PilImage:
    """
    Create a pretty image of a molecule,
    with a colored frame around it

    :param mol: the molecule
    :param frame_color: the color of the frame
    :param size: the size of the image
    :return: the produced image
    """
    img = Draw.MolToImage(mol, size=(size, size))
    cropped_img = crop_image(img)
    return draw_rounded_rectangle(cropped_img, frame_color)


def molecules_to_images(
    mols: Sequence[Chem.rdchem.Mol],
    frame_colors: Sequence[PilColor],
    size: int = 300,
    draw_kwargs: Dict[str, Any] = None,
) -> List[PilImage]:
    """
    Create pretty images of molecules with a colored frame around each one of them.

    The molecules will be resized to be of similar sizes.

    :param smiles_list: the molecules
    :param frame_colors: the color of the frame for each molecule
    :param size: the sub-image size
    :param draw_kwargs: additional keyword-arguments sent to `MolsToGridImage`
    :return: the produced images
    """
    draw_kwargs = draw_kwargs or {}
    all_mols = Draw.MolsToGridImage(mols, molsPerRow=len(mols), subImgSize=(size, size), **draw_kwargs)
    if not hasattr(all_mols, "crop"):  # Is not a PIL image
        fileobj = io.BytesIO(all_mols.data)
        all_mols = Image.open(fileobj)

    images = []
    for idx, frame_color in enumerate(frame_colors):
        image_obj = all_mols.crop((size * idx, 0, size * (idx + 1), size))
        image_obj = crop_image(image_obj)
        images.append(draw_rounded_rectangle(image_obj, frame_color))
    return images


def crop_image(img: PilImage, margin: int = 20) -> PilImage:
    """
    Crop an image by removing white space around it

    :param img: the image to crop
    :param margin: padding, defaults to 20
    :return: the cropped image
    """
    # pylint: disable=invalid-name
    # First find the boundaries of the white area
    x0_lim = img.width
    y0_lim = img.height
    x1_lim = 0
    y1_lim = 0
    for x in range(0, img.width):
        for y in range(0, img.height):
            if img.getpixel((x, y)) != (255, 255, 255):
                if x < x0_lim:
                    x0_lim = x
                if x > x1_lim:
                    x1_lim = x
                if y < y0_lim:
                    y0_lim = y
                if y > y1_lim:
                    y1_lim = y
    x0_lim = max(x0_lim, 0)
    y0_lim = max(y0_lim, 0)
    x1_lim = min(x1_lim + 1, img.width)
    y1_lim = min(y1_lim + 1, img.height)
    # Then crop to this area
    cropped = img.crop((x0_lim, y0_lim, x1_lim, y1_lim))
    # Then create a new image with the desired padding
    out = Image.new(
        img.mode,
        (cropped.width + 2 * margin, cropped.height + 2 * margin),
        color="white",
    )
    out.paste(cropped, (margin + 1, margin + 1))
    return out


def draw_rounded_rectangle(img: PilImage, color: PilColor, arc_size: int = 20) -> PilImage:
    """
    Draw a rounded rectangle around an image

    :param img: the image to draw upon
    :param color: the color of the rectangle
    :param arc_size: the size of the corner, defaults to 20
    :return: the new image
    """
    # pylint: disable=invalid-name
    x0, y0, x1, y1 = img.getbbox()
    x1 -= 1
    y1 -= 1
    copy = img.copy()
    draw = ImageDraw.Draw(copy)
    arc_size_half = arc_size // 2
    draw.arc((x0, y0, arc_size, arc_size), start=180, end=270, fill=color)
    draw.arc((x1 - arc_size, y0, x1, arc_size), start=270, end=0, fill=color)
    draw.arc((x1 - arc_size, y1 - arc_size, x1, y1), start=0, end=90, fill=color)
    draw.arc((x0, y1 - arc_size, arc_size, y1), start=90, end=180, fill=color)
    draw.line((x0 + arc_size_half, y0, x1 - arc_size_half, y0), fill=color)
    draw.line((x1, arc_size_half, x1, y1 - arc_size_half), fill=color)
    draw.line((arc_size_half, y1, x1 - arc_size_half, y1), fill=color)
    draw.line((x0, arc_size_half, x0, y1 - arc_size_half), fill=color)
    return copy


class RouteImageFactory:
    """
    Factory class for drawing a route

    :param route: the dictionary representation of the route
    :param in_stock_colors: the colors around molecules, defaults to {True: "green", False: "orange"}
    :param show_all: if True, also show nodes that are marked as hidden
    :param margin: the margin between images
    :param mol_size: the size of the molecule
    :param mol_draw_kwargs: additional arguments sent to the drawing routine
    :param replace_mol_func: an optional function to replace molecule images
    """

    def __init__(
        self,
        route: Dict[str, Any],
        in_stock_colors: FrameColors = None,
        show_all: bool = True,
        margin: int = 100,
        mol_size: int = 300,
        mol_draw_kwargs: Dict[str, Any] = None,
        replace_mol_func: Callable[[Dict[str, Any]], None] = None,
    ) -> None:
        in_stock_colors = in_stock_colors or {
            True: "green",
            False: "orange",
        }
        self.show_all: bool = show_all
        self.margin: int = margin

        self._stock_lookup: Dict[str, Any] = {}
        self._mol_lookup: Dict[str, Any] = {}
        self._extract_molecules(route)
        if replace_mol_func is not None:
            replace_mol_func(self._mol_lookup)
        images = molecules_to_images(
            list(self._mol_lookup.values()),
            [in_stock_colors[val] for val in self._stock_lookup.values()],
            size=mol_size,
            draw_kwargs=mol_draw_kwargs or {},
        )
        self._image_lookup = dict(zip(self._mol_lookup.keys(), images))

        self._mol_tree = self._extract_mol_tree(route)
        self._add_effective_size(self._mol_tree)

        pos0 = (
            self._mol_tree["eff_width"] - self._mol_tree["image"].width + self.margin,
            int(self._mol_tree["eff_height"] * 0.5) - int(self._mol_tree["image"].height * 0.5),
        )
        self._add_pos(self._mol_tree, pos0)

        self.image = Image.new(
            self._mol_tree["image"].mode,
            (self._mol_tree["eff_width"] + self.margin, self._mol_tree["eff_height"]),
            color="white",
        )
        self._draw = ImageDraw.Draw(self.image)
        self._make_image(self._mol_tree)
        self.image = crop_image(self.image)

    def _add_effective_size(self, tree_dict: Dict[str, Any]) -> None:
        children = tree_dict.get("children", [])
        for child in children:
            self._add_effective_size(child)
        if children:
            tree_dict["eff_height"] = sum(child["eff_height"] for child in children) + self.margin * (len(children) - 1)
            tree_dict["eff_width"] = (
                max(child["eff_width"] for child in children) + tree_dict["image"].size[0] + self.margin
            )
        else:
            tree_dict["eff_height"] = tree_dict["image"].size[1]
            tree_dict["eff_width"] = tree_dict["image"].size[0] + self.margin

    def _add_pos(self, tree_dict: Dict[str, Any], pos: Tuple[int, int]) -> None:
        tree_dict["left"] = pos[0]
        tree_dict["top"] = pos[1]
        children = tree_dict.get("children")
        if not children:
            return

        mid_y = pos[1] + int(tree_dict["image"].height * 0.5)  # Mid-point of image along y
        children_height = sum(child["eff_height"] for child in children) + self.margin * (len(children) - 1)
        childen_leftmost = min(pos[0] - self.margin - child["image"].width for child in children)
        child_y = mid_y - int(children_height * 0.5)  # Top-most edge of children
        child_ys = []
        # Now compute first guess of y-pos for children
        for child in children:
            y_adjust = int((child["eff_height"] - child["image"].height) * 0.5)
            child_ys.append(child_y + y_adjust)
            child_y += self.margin + child["eff_height"]

        for idx, (child, child_y0) in enumerate(zip(children, child_ys)):
            child_x = childen_leftmost  # pos[0] - self.margin - child["image"].width
            child_y = child_y0
            # Overwrite first guess if child does not have any children
            if not child.get("children") and idx == 0 and len(children) > 1:
                child_y = child_ys[idx + 1] - self.margin - child["image"].height
            elif not child.get("children") and idx > 0:
                child_y = child_ys[idx - 1] + self.margin + children[idx - 1]["image"].height
            self._add_pos(child, (child_x, child_y))

    def _extract_mol_tree(self, tree_dict: Dict[str, Any]) -> Dict[str, Any]:
        dict_ = {
            "smiles": tree_dict["smiles"],
            "image": self._image_lookup[tree_dict["smiles"]],
        }
        if tree_dict.get("children"):
            dict_["children"] = [
                self._extract_mol_tree(grandchild)
                for grandchild in tree_dict.get("children")[0]["children"]  # type: ignore
                if not (grandchild.get("hide", False) and not self.show_all)
            ]
        return dict_

    def _extract_molecules(self, tree_dict: Dict[str, Any]) -> None:
        if tree_dict["type"] == "mol":
            self._stock_lookup[tree_dict["smiles"]] = tree_dict.get("in_stock", False)
            self._mol_lookup[tree_dict["smiles"]] = Chem.MolFromSmiles(tree_dict["smiles"])
        for child in tree_dict.get("children", []):
            self._extract_molecules(child)

    def _make_image(self, tree_dict: Dict[str, Any]) -> None:
        self.image.paste(tree_dict["image"], (tree_dict["left"], tree_dict["top"]))
        children = tree_dict.get("children")
        if not children:
            return

        children_right = max(child["left"] + child["image"].width for child in children)
        mid_x = children_right + int(0.5 * (tree_dict["left"] - children_right))
        mid_y = tree_dict["top"] + int(tree_dict["image"].height * 0.5)

        self._draw.line((tree_dict["left"], mid_y, mid_x, mid_y), fill="black")
        for child in children:
            self._make_image(child)
            child_mid_y = child["top"] + int(0.5 * child["image"].height)
            self._draw.line(
                (
                    mid_x,
                    mid_y,
                    mid_x,
                    child_mid_y,
                    child["left"] + child["image"].width,
                    child_mid_y,
                ),
                fill="black",
            )
        self._draw.ellipse((mid_x - 8, mid_y - 8, mid_x + 8, mid_y + 8), fill="black", outline="black")
