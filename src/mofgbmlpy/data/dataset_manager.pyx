from mofgbmlpy.data.input_density import Input_density


class TrainDatasetManager:
    def __init__(self, base_dir, identifier, node_range=(10, 75)):
        """
        トレインデータセットを管理するクラス

        Args:
            base_dir (Path): トレインデータのベースディレクトリ
            identifier (str): tstファイルから取得した識別子
            node_range (tuple): フィルタリングするノード番号の範囲 (デフォルト: 10～75)
        """
        self.base_dir = Path(base_dir)
        self.identifier = identifier
        self.node_range = node_range
        self.train_files = self._load_train_files()

    def _load_train_files(self):
        """
        対応するトレインファイルをロードして条件に基づきフィルタリング

        Returns:
            dict: {an_m_key: [Path]} の形式で条件ごとにグループ化されたトレインファイル
        """
        train_dir = self.base_dir / f"{self.identifier}_tra"
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

        # 条件に基づいてファイルを探索
        train_files = train_dir.glob(f"{self.identifier}_node*.csv")

        # an_m をキーとしてファイルをグループ化
        grouped_files = {}
        for file in train_files:
            # ファイル名から `nodeXX` を抽出してノード番号でフィルタリング
            node_number = int(re.search(r"node(\d+)", file.stem).group(1))
            if not (self.node_range[0] <= node_number <= self.node_range[1]):
                continue

            # an_m を抽出してグループ化
            an_m_match = re.search(r"an_m(\d+)", file.stem)
            if not an_m_match:
                continue
            an_m_key = an_m_match.group(1)

            if an_m_key not in grouped_files:
                grouped_files[an_m_key] = []
            grouped_files[an_m_key].append(file)

        # 各グループをノード番号順にソート
        for key in grouped_files:
            grouped_files[key] = sorted(grouped_files[key], key=lambda x: int(re.search(r"node(\d+)", x.stem).group(1)))

        return grouped_files

    def get_datasets_by_an_m(self, an_m_key):
        """
        指定された an_m に対応するトレインデータセットを取得

        Args:
            an_m_key (str): グループ化キー (例: "1", "2", ...)

        Returns:
            list: 指定された条件に対応するデータセットリスト
        """
        if an_m_key not in self.train_files:
            raise ValueError(f"No datasets found for an_m: {an_m_key}")

        return [Input_density().input_data_set(file, False) for file in self.train_files[an_m_key]]

    def get_all_an_m_keys(self):
        """
        利用可能な an_m のリストを取得

        Returns:
            list: 利用可能な an_m のキーリスト
        """
        return list(self.train_files.keys())
