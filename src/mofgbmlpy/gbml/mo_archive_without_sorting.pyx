from pymoo.util.archive import MultiObjectiveArchive

class MoArchiveWithoutSorting(MultiObjectiveArchive):
    def __new__(cls, max_size=200, truncate_size=100, **kwargs):
        return super().__new__(cls,
                               max_size=max_size,
                               truncate_size=truncate_size,
                               **kwargs)

    def _find_opt(self, sols):
        return sols