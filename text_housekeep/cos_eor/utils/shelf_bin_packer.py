from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from dataclasses import dataclass
from enum import Enum
import math

import magnum as mn

MIN_SHELF_HEIGHT = 0.01
PAD_SHELF = 0.002
PAD_RECT_ORIGIN = PAD_SHELF / 2
NP_ABS_TOL = 1e-06
OCCP = "occupied"
FREE = "free"


def area(range: mn.Range2D):
    return range.size_x() * range.size_y()


class Rect(object):
    def __init__(self, id: int, rect_range: mn.Range2D, is_rotated=False):
        self.id = id
        self.range = rect_range
        self.is_rotated = is_rotated

    def copy(self):
        return Rect(
            self.id,
            mn.Range2D(self.range.bottom_left, self.range.top_right),
            self.is_rotated)

    def rotate(self):
        rotated_rect = self.copy()
        rotated_rect.range = mn.Range2D.from_size(
            rotated_rect.range.bottom_left,
            rotated_rect.range.size().flipped()
        )
        rotated_rect.is_rotated = True
        return rotated_rect

    def __repr__(self):
        return (
            f"Rect(id={self.id}, "
            f"bot_left={self.range.bottom_left}, "
            f"size={self.range.size()}, "
            f"is_rotated={self.is_rotated})"
        )

    def to_dict(self, keep="id", id_to_key=None):
        return {
            "id": self.id if keep == "id" else id_to_key[self.id],
            "size": [self.range.size_x(), self.range.size_y()],
            "bottom_left": [self.range.left, self.range.bottom],
            "is_rotated": self.is_rotated,
        }

    @classmethod
    def from_dict(cls, info, key_to_id=None):
        rect_range = mn.Range2D.from_size(mn.Vector2(info["bottom_left"]), mn.Vector2(info["size"]))
        if key_to_id is not None:
            return cls(key_to_id[info["id"]], rect_range, info["is_rotated"])
        else:
            return cls(info["id"], rect_range, info["is_rotated"])


@dataclass
class Match:
    """Store properties of every object placed in a shelf"""
    rect: Rect
    score: float
    space_idx: int
    shelf_idx: int

    def to_dict(self, keep="id", id_to_key=None):
        return {
            "rect": self.rect.to_dict(keep, id_to_key),
            "score": self.score,
            "space_idx": int(self.space_idx),
            "shelf_idx": int(self.shelf_idx)
        }

    @classmethod
    def from_dict(cls, info, key_to_id=None):
        rect = Rect.from_dict(info["rect"], key_to_id)
        return cls(rect, info["score"], info["space_idx"], info["shelf_idx"])


@dataclass
class Space:
    kind: str  # free or occ
    dims: mn.Range1D

    def to_dict(self):
        return {
            "kind": self.kind,
            "dims": [self.dims.min, self.dims.max]
        }


# lower score is better!
class ScoringFuncs(Enum):
    REDUCE_AREA = 0
    MINIMIZE_HEIGHT_DIFF = 1
    MINIMIZE_WIDTH_DIFF = 1
    MINIMIZE_WASTE_SPACE = 2


class Shelf(object):
    def __init__(self, dims: mn.Range2D, max_height: float, idx: int, spaces:List[Space]=None):
        # magnum 2D range
        self.dims = dims
        # maximum height you can stretch the current shelf(= receptacle/current height)
        self.max_height = max_height
        # list of magnum 1D ranges
        if spaces is None:
            self.spaces = [Space(FREE, self.dims.x())]
        else:
            self.spaces = spaces
        # try to place in a space where width is closest to rectangle
        self.scoring_function = ScoringFuncs.MINIMIZE_HEIGHT_DIFF
        # index within packer
        self.idx = idx

    def to_dict(self):
        return {
            # add dims
            "size": [self.dims.size_x(), self.dims.size_y()],
            "bottom_left": [self.dims.left, self.dims.bottom],
            "max_height": self.max_height,
            # add spaces
            "spaces": [space.to_dict() for space in self.spaces],
            # add shelf index
            "idx": self.idx
        }

    @classmethod
    def from_dict(cls, info):
        dims = mn.Range2D.from_size(mn.Vector2(info["bottom_left"]), mn.Vector2(info["size"]))
        spaces = [Space(space["kind"], mn.Range1D(space["dims"])) for space in info["spaces"]]
        return cls(dims, info["max_height"], info["idx"], spaces)

    @property
    def free_area(self):
        return sum([space.dims.size() * self.dims.size_y() for space in self.spaces if space.kind == FREE])

    @property
    def occp_area(self):
        return sum([space.dims.size() * self.dims.size_y() for space in self.spaces if space.kind == OCCP])

    def assert_consistency(self, num_shelves):
        # total area must be conserved at all times
        if not np.isclose(self.free_area + self.occp_area, area(self.dims), atol=NP_ABS_TOL):
            import pdb
            pdb.set_trace()

        # last shelf cannot exist without occupied area
        if np.isclose(self.occp_area, 0, atol=NP_ABS_TOL) and self.idx == num_shelves-1:
            import pdb
            pdb.set_trace()

        # assert no free-spaces are unmerged
        self.assert_spaces_consistency()

    def assert_spaces_consistency(self):
        # assert no free-spaces are unmerged
        for prev, next in zip(self.spaces[:-1], self.spaces[1:]):
            if prev.kind == next.kind == FREE:
                import pdb
                pdb.set_trace()
            if not np.isclose(prev.dims.max, next.dims.min):
                import pdb
                pdb.set_trace()

    def _score(self, rect: Rect, shelf_height: float, space_width: float):
        # lower score is better!
        if self.scoring_function == ScoringFuncs.REDUCE_AREA:
            # width x height_extended by current rect (used)
            extra_free_area = self.dims.size_x() * (shelf_height - self.dims.size_y())
            return self.free_area + extra_free_area - area(rect.range)
        # Todo: Fix later
        elif self.scoring_function == ScoringFuncs.MINIMIZE_HEIGHT_DIFF:
            # height difference between current free-space and rect
            return abs(self.dims.size_y() - rect.range.size_y())

        elif self.scoring_function == ScoringFuncs.MINIMIZE_WIDTH_DIFF:
            # width difference between free-space and rect
            return abs(space_width - rect.range.size_x())

        # elif self.scoring_function == ScoringFuncs.MINIMIZE_WASTE_SPACE:
        #     shelf_height_diff = shelf_height - self.dims.size_y()
        #     extra_waste_area = self.occupied_length * (shelf_height_diff)
        #     return rect.range.size_x() * shelf_height_diff + extra_waste_area
        raise AssertionError

    def _best_match(self, rect: Rect) -> Tuple[Match, float]:
        """Iterate over all free-spaces in this shelf and return the most efficient one"""
        best_match = Match(None, math.inf, -1, -1)
        shelf_height = self.dims.size_y()
        # if rect height more than current shelf_height
        if rect.range.size_y() > shelf_height:
            if rect.range.size_y() <= self.max_height:
                # if rect height less than max possible shelf height
                shelf_height = min(rect.range.size_y() + PAD_SHELF, self.max_height)
            else:
                # otherwise we can't place
                return best_match, shelf_height

        # try to insert rect in one of the free-spaces here
        for space_idx, space in enumerate(self.spaces):
            if space.kind == OCCP:
                continue

            free_space = mn.Range2D.from_size(
                mn.Vector2(space.dims.min, self.dims.bottom),
                mn.Vector2(space.dims.size(), shelf_height)
            )

            # align the rectangle bottom-left within free-space with some origin padding
            moved_rect = rect.copy()
            # look for slightly more free-space along the row than exactly needed
            moved_rect.range = moved_rect.range.padded(mn.Vector2(PAD_SHELF, 0))
            translation = free_space.bottom_left - moved_rect.range.bottom_left
            translation = mn.Vector2(PAD_RECT_ORIGIN, PAD_RECT_ORIGIN) + translation
            moved_rect.range = rect.range.translated(translation)

            # check if it can be placed in free-space and score it
            if not free_space.contains(moved_rect.range):
                continue
            score = self._score(moved_rect, shelf_height, space.dims.size())
            assert score >= 0

            # compare with best-score until now
            if score < best_match.score:
                best_match.rect = moved_rect
                best_match.score = score
                best_match.space_idx = space_idx

        return best_match, shelf_height

    def merge_free_spaces(self, matches):
        def get_consecutive(inds):
            from itertools import groupby
            from operator import itemgetter
            ranges = []
            for k, g in groupby(enumerate(inds), lambda ix: ix[0] - ix[1]):
                ranges.append(list(map(itemgetter(1), g)))
            return ranges

        # detect consecutive free spaces
        free_space_inds = [idx for idx, space in enumerate(self.spaces) if space.kind == FREE]
        free_space_ranges = get_consecutive(free_space_inds)

        idx = 0
        offset_inds = np.arange(0, len(self.spaces))
        new_spaces = []

        # process each range and all indices in between
        for _range in free_space_ranges:
            rlow, rhigh = _range[0], _range[-1]

            # append all occupied spaces in between
            while rlow > idx:
                assert self.spaces[idx].kind == OCCP
                new_spaces.append(self.spaces[idx])
                idx += 1

            # merge spaces and add -- all the space between is covered
            joined_free_space = mn.Range1D(self.spaces[rlow].dims.min, self.spaces[rhigh].dims.max)

            new_spaces.append(Space(FREE, joined_free_space))

            # add offset caused by the merging
            merge_offset = rhigh - rlow
            offset_inds[rhigh+1:] = offset_inds[rhigh+1:] - merge_offset

            # increase counter
            idx = rhigh + 1

        # account for trailing occupied spaces
        while len(self.spaces) > idx:
            assert self.spaces[idx].kind == OCCP
            new_spaces.append(self.spaces[idx])
            idx += 1

        # replace with merged spaces
        # old_spaces = deepcopy([space.to_dict() for space in self.spaces])
        self.spaces = new_spaces

        # check consistency
        self.assert_spaces_consistency()

        # adjust offsets across matches
        for _, match in matches.items():
            if match.shelf_idx == self.idx:
                match.space_idx = offset_inds[match.space_idx]
        return matches

    def add_rect(self, match: Match, new_shelf_height: float, matches: Dict[int, Match]):
        rect = match.rect
        free_space_idx = match.space_idx

        try:
            assert self.spaces[free_space_idx].kind == FREE
            assert self.dims.size_y() <= new_shelf_height <= self.max_height
            assert rect.range.size_y() <= new_shelf_height
            assert 0 <= free_space_idx <= len(self.spaces)
        except:
            import pdb
            pdb.set_trace()

        self.dims.top = self.dims.bottom + new_shelf_height

        # slice the free-space into occupied and free space
        free_space = self.spaces.pop(free_space_idx)
        slice_point = min(rect.range.right + PAD_SHELF, self.dims.right)

        # insert new occupied space
        new_occp_space = mn.Range1D(free_space.dims.min, slice_point)
        occ_space_idx = free_space_idx
        self.spaces.insert(occ_space_idx, Space(OCCP, new_occp_space))

        # store occupied-space index in match;
        match.space_idx = occ_space_idx

        # insert new free space
        new_free_space = mn.Range1D(slice_point, free_space.dims.max)
        free_space_idx = free_space_idx + 1
        self.spaces.insert(free_space_idx, Space(FREE, new_free_space))

        # adjust offsets across matches ahead
        for _, match in matches.items():
            if match.shelf_idx == self.idx and match.space_idx >= free_space_idx:
                match.space_idx += 1
        return matches

    def insert_check(self, rect: Rect) -> Tuple[Match, float]:
        """We should check both rotations because any one of them might be more suitable."""
        # insert
        best_match, new_shelf_height = self._best_match(rect)

        # rotate and insert
        rrect = rect.rotate()
        rbest_match, rnew_shelf_height = self._best_match(rrect)
        if rbest_match.space_idx != -1:
            rbest_match.rect.is_rotated = True

        # compare scores to check efficiency and feasibility of both
        if best_match.score < rbest_match.score:
            match, shelf_height = best_match, new_shelf_height
        else:
            match, shelf_height = rbest_match, rnew_shelf_height

        if match.space_idx != -1:
            return match, shelf_height
        else:
            return None, None

    def remove(self, match: Match):
        # just replace occupied space with free space, we merge spaces in packer next
        self.spaces[match.space_idx].kind = FREE


class ShelfBinPacker(object):
    """Store a list of shelves going from bottom to top"""

    def __init__(self, dims: mn.Range2D):
        # magnum 2D range
        self.dims = dims
        self.shelves: List[Shelf] = []
        self.matches: Dict[int, Match] = {}

    def assert_consistency(self):
        # assert consistency in shelf organization
        for prev, next in zip(self.shelves[:-1], self.shelves[1:]):
            # assert shelves are organized from bottom to top
            if prev.dims.top >= next.dims.top or prev.dims.bottom >= next.dims.bottom:
                import pdb
                pdb.set_trace()
            # assert shelves span same width
            if not np.isclose(prev.dims.left, next.dims.left, atol=NP_ABS_TOL) \
                    or not np.isclose(prev.dims.right, next.dims.right, atol=NP_ABS_TOL):
                import pdb
                pdb.set_trace()

        # assert consistency on each shelf
        for shelf in self.shelves:
            shelf.assert_consistency(num_shelves=len(self.shelves))

        # assert all the space indices in matches are occupied
        for match in self.matches.values():
            if self.shelves[match.shelf_idx].spaces[match.space_idx].kind != OCCP:
                import pdb
                pdb.set_trace()

    def _add_shelf(self, shelf_height):
        """Add a new shelf (row) on top"""
        if shelf_height < MIN_SHELF_HEIGHT:
            return None

        if len(self.shelves) == 0:
            # if packer is empty
            shelf_dims = mn.Range2D.from_size(
                self.dims.bottom_left,
                mn.Vector2(self.dims.size_x(), shelf_height)
            )
            max_shelf_height = self.dims.size_y()
            if shelf_height > max_shelf_height:
                return None
        else:
            # if there are other shelves, pick the last one
            last_shelf = self.shelves[-1]
            max_shelf_height = self.dims.top - last_shelf.dims.top
            if shelf_height > max_shelf_height:
                return None
            last_shelf.max_height = last_shelf.dims.size_y()
            shelf_dims = mn.Range2D.from_size(
                last_shelf.dims.top_left,
                mn.Vector2(self.dims.size_x(), shelf_height)
            )
        shelf = Shelf(shelf_dims, max_shelf_height, len(self.shelves))
        return shelf

    def insert(self, rect: Rect):
        best_match = Match(None, math.inf, -1, -1)
        best_shelf_new_height = 0

        # iterate over all existing shelves from bottom to top
        for shelf_idx, shelf in enumerate(self.shelves):
            # try inserting in each shelf
            match, new_shelf_height = shelf.insert_check(rect)
            if match is None:
                continue
            # update best shelf found until now
            elif match.score < best_match.score:
                best_match = match
                best_match.shelf_idx = shelf_idx
                best_shelf_new_height = new_shelf_height

        # if couldn't place in any existing shelf
        if best_match.shelf_idx == -1:
            # try inserting a new shelf
            new_shelf = self._add_shelf(min(rect.range.size_y(), rect.range.size_x()) + PAD_SHELF)
            # cant insert shelf return None
            if new_shelf is None:
                return None
            best_match, best_shelf_new_height = new_shelf.insert_check(rect)
            # cant place on shelf then return None
            if best_match is None:
                return None
            # add new shelf to packer
            else:
                best_match.shelf_idx = len(self.shelves)
                self.shelves.append(new_shelf)

        # store the object in packer by it's object-id
        self.matches[best_match.rect.id] = best_match

        # add the object to the best shelf and space
        self.matches = self.shelves[best_match.shelf_idx].add_rect(best_match, best_shelf_new_height, self.matches)

        # merge all free spaces on the shelf edited (likely don't need this)
        self.matches = self.shelves[best_match.shelf_idx].merge_free_spaces(self.matches)

        # final assert
        self.assert_consistency()

        return best_match

    def _remove_empty_shelves(self):
        """Remove empty shelves from the end of the shelf list"""
        delete_start_idx = len(self.shelves)

        # iterate over each shelf from bottom to top
        for shelf_idx in range(len(self.shelves) - 1, -1, -1):
            shelf = self.shelves[shelf_idx]
            # compute that shelf's area
            shelf_area = area(shelf.dims)
            # if they're close that means the entire shelf is empty
            if np.isclose(shelf_area, shelf.free_area, atol=NP_ABS_TOL):
                delete_start_idx = shelf_idx
            else:
                break
        # delete this shelf and everything below
        del self.shelves[delete_start_idx:]

    def remove(self, rect_id: int) -> bool:
        # return if match(rect) not in packer
        if rect_id not in self.matches:
            return False

        # get shelf from match
        rect_match = self.matches[rect_id]
        shelf_idx = rect_match.shelf_idx
        shelf = self.shelves[shelf_idx]

        # remove rect from shelf and matches
        shelf.remove(rect_match)
        del self.matches[rect_id]

        # merge all free spaces on the shelf edited
        self.shelves[rect_match.shelf_idx].merge_free_spaces(self.matches)

        # remove empty shelves if we remove rect from last-shelf
        if shelf_idx == len(self.shelves) - 1:
            self._remove_empty_shelves()

        # final assert
        self.assert_consistency()

        return True

    def get_objs(self):
        return list(self.matches.keys())

    def to_dict(self, keep="id", id_to_key=None):
        # final assert
        self.assert_consistency()

        state = {
            "shelves": [shelf.to_dict() for shelf in self.shelves],
            "bottom_left": [self.dims.left, self.dims.bottom],
            "size": [self.dims.size_x(), self.dims.size_y()],
            "matches": [match.to_dict(keep, id_to_key) for match in self.matches.values()]
        }
        return state

    def from_dict(self, state, key_to_id=None):
        # assert right receptacle dims
        try:
            assert all(np.isclose([self.dims.left, self.dims.bottom], state["bottom_left"], atol=NP_ABS_TOL))
            assert all(np.isclose([self.dims.size_x(), self.dims.size_y()], state["size"], atol=NP_ABS_TOL))
        except:
            import pdb
            pdb.set_trace()

        # build shelves
        self.shelves = [Shelf.from_dict(shelf_state) for shelf_state in state["shelves"]]

        # build matches
        self.matches = [Match.from_dict(match_state, key_to_id) for match_state in state["matches"]]
        self.matches = {match.rect.id:match for match in self.matches}

        # build range
        self.dims = mn.Range2D.from_size(mn.Vector2(state["bottom_left"]), mn.Vector2(state["size"]))

        # final assert
        self.assert_consistency()
