import pandas
import numpy
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


def is_number(s):
    """
    judge if the input s is a number

    for example:
    input '3', return True
    input '1.2' return True
    input '1.a3' return False
    input '四' return True
    :param s:
    :return:
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


class excelProcessor(object):
    """
    This class is used to process excel file or dataframe object
    """

    def __init__(self, filename):
        self.name = filename  # 处理的文件名
        self.df = None  #
        if isinstance(filename, list):  # 根据初始化文件类型使用不同的方法构造dataframe
            self.df = pandas.DataFrame(filename)
        else:
            self.df = pandas.read_excel(filename)
        self.df_matrix = self.df.values  # 将dataframe转为numpy.ndarray形式,返回DataFrame的Numpy表示形式
        self.df_rows, self.df_cols = self.df_matrix.shape  # 记录numpy.ndarray的行列数目,即返回一个表示DataFrame维数的元组

        self.col_start = 0  # 记录从数值形式的值从哪一列开始
        # 最初考虑excel文件中不只包含特征数据值，还包括一些描述信息(目前已取消)
        # 预设表格的第一列是编号，第二列开始为字符串形式的描述信息，某一列开始为特征数据信息(目前已取消)
        for i in range(self.df_cols):  # 该循环是为寻找到数据信息开始的列数
            # print(type(df_matrix[0][i]))
            if isinstance(self.df_matrix[0][i], float):
                self.col_start = i
                break

        self.dfvalue = self.df.iloc[:, self.col_start:]  # 记录特征数据信息的dataframe
        self.valuearray = self.dfvalue.values  # 将特征数据信息转为numpy.ndarray形式
        self.col_name = [column_name for column_name in self.df][self.col_start:]  # 记录特征名,直接遍历获取列名

        self.dfattr = self.df.iloc[:, self.col_start:-1]  # 记录独立属性dataframe
        # 注：独立属性指的是可以根据这些属性预测某个属性的值，类似于自变量。相似地，决策属性类似于应变量
        self.attr_num = len([i for i in self.dfattr])  # 记录独立属性数目
        self.dftarget = self.df.iloc[:, -1]  # 记录决策属性dataframe

        self.count = numpy.shape(self.dfattr)[0]  # 重复定义。记录样本数目
        self.dim = numpy.shape(self.dfattr)[1]  # 重复定义。记录特征数目
        # self.X = preprocessing.MinMaxScaler().fit_transform(self.dfattr.values)
        self.X = self.dfattr.values  # 重复定义。记录独立属性numpy.ndarray
        self.Y = self.dftarget.values  # 重复定义。记录决策属性numpy.ndarray

    def get_column_names(self):
        """

        :return: excel表格或dataframe列名
        """
        return [column_name for column_name in self.df]

    def get_data(self):
        """
        get data from excel by rows
        :return: a list containing each row in dict (column names are keys)
        """
        materials = []

        for row in self.df.iterrows():
            material = {}

            for column_name in self.get_column_names():
                material[column_name.replace('.', '').replace('$', '')] = row[1][column_name]

            materials.append(material)

        return materials

    def has_blank_cell(self):
        """

        :return: 若表格或dataframe中有空值，则返回true，否则返回false
        """
        if 0 == numpy.where(self.df.isnull())[0].size:
            return False
        return True

    def blank_cells(self):
        """

        :return: 返回表格或dataframe中空值的坐标
        """
        blank_matrix = numpy.where(self.df.isnull())
        return list(zip(blank_matrix[0], blank_matrix[1]))

    def statistics_data_check(self):
        """
        计算数据的统计信息
        :return:
        """
        geo_mean = []
        # 求每列的几何平均
        for i in range(len(self.col_name)):
            col_i = self.valuearray[:, i]
            prod = 1.0
            for j in col_i:
                prod *= j
            geo_mean.append(pow(prod, 1.0 / self.df_rows))

        desc = self.dfvalue.describe()  # 特征数据的统计描述
        skew = self.dfvalue.skew()  # 特征数据的偏度
        rangem = desc.loc['max'] - desc.loc['min']  # 特征数据的极差

        IQR = desc.loc['75%'] - desc.loc['25%']  # 特征数据的四分位距
        upper = desc.loc['75%'] + 1.5 * IQR  # 特征数据的盒图上限
        lower = desc.loc['25%'] - 1.5 * IQR  # 特征数据的盒图下限

        # 将上述信息融入统计描述中
        desc.loc['range'] = rangem
        desc.loc['IQR'] = IQR
        desc.loc['lower'] = lower
        desc.loc['upper'] = upper
        desc.loc['geomean'] = geo_mean
        desc.loc['skew'] = skew

        return desc

    def eudist_data_check(self):
        """
        相似样本对比分析
        :return:
        """
        df_arg = self.df.iloc[:, self.col_start:-1]
        df_func = self.df.iloc[:, -1]

        arg_mat = df_arg.values  # 非决策属性值矩阵
        func_mat = df_func.values.reshape(-1, 1)  # 决策属性值矩阵

        arg_pair_dist = pairwise_distances(arg_mat)  # 非决策属性距离矩阵
        func_pair_dist = pairwise_distances(func_mat)  # 决策属性距离矩阵

        # turn pair dist matrix to list
        arg_dist_list = [arg_pair_dist[i][j] for i in range(self.df_rows) for j in range(i + 1, self.df_rows)]
        func_dist_list = [func_pair_dist[i][j] for i in range(self.df_rows) for j in range(i + 1, self.df_rows)]

        # 属性上下限，判断样本属性是否相似或差异
        arg_dist_upper = numpy.quantile(arg_dist_list, 0.75)
        arg_dist_lower = numpy.quantile(arg_dist_list, 0.25)
        func_dist_upper = numpy.quantile(func_dist_list, 0.9)
        func_dist_lower = numpy.quantile(func_dist_list, 0.1)

        suspected_samples = []  # 记录嫌疑样本
        suspected_samples_pair = []  # 记录嫌疑样本对, 即两个规则差异大的样本
        suspected_samples_count = {}  # 记录嫌疑样本出现次数

        for i in range(self.df_rows):
            for j in range(i + 1, self.df_rows):
                #  如果样本非决策属性差异大而决策属性相似，记入嫌疑样本
                if arg_pair_dist[i][j] >= arg_dist_upper and func_pair_dist[i][j] <= func_dist_lower:
                    suspected_samples.append(i)
                    suspected_samples.append(j)
                    suspected_samples_pair.append((i, j))
                #  如果样本非决策属性相似而决策属性差异大，记入嫌疑样本
                if arg_pair_dist[i][j] <= arg_dist_lower and func_pair_dist[i][j] >= func_dist_upper:
                    suspected_samples.append(i)
                    suspected_samples.append(j)
                    suspected_samples_pair.append((i, j))

        for i in suspected_samples:
            # 若嫌疑样本出现3次以上，视为异常样本
            if suspected_samples.count(i) > 1:
                suspected_samples_count[i] = suspected_samples.count(i)
        suspected_samples_count = sorted(suspected_samples_count.items(), key=lambda it: it[1], reverse=True)[
                                  :int(self.df_rows * 0.1)]
        # {样本编号：出现次数}
        return dict(suspected_samples_count)

    def algorithm_data_check(self):
        """
        局部异常因子检测、孤立森林检测离群点
        :return:
        """
        lof = LocalOutlierFactor(n_neighbors=self.df_rows // 2, contamination=.1)
        llof = lof.fit_predict(self.valuearray)
        llof_idx = [i for i in range(len(llof)) if llof[i] == -1]
        iif = IsolationForest(n_estimators=len(self.col_name) * 2, contamination=.1)
        lif = iif.fit_predict(self.valuearray)
        lif_idx = [i for i in range(len(lif)) if lif[i] == -1]
        dbs = DBSCAN(eps=0.3, min_samples=5).fit_predict(self.valuearray[:, :2])
        dbs_idx = [i for i in range(len(dbs)) if dbs[i] == -1]
        desc = {}
        desc['lof'] = llof_idx
        desc['if'] = lif_idx
        desc['dbs'] = dbs_idx
        return desc

    def get_corr_coef(self, method):
        """
        计算维度间的相关系数，返回相关系数较高的特征对
        :param method: 何种相关系数（Pearson，Spearman，Kendall）
        新增判定系数、点二列相关
        :return:
        """
        flag = 0
        attr_relate = []  # 记录相关系数过高的特征编号和相关系数，例如：(1,2,0.9)表示特征1与特征2的相关系数为0.9
        if method == 'coef_determination':
            method = 'pearson'
            flag = 1
        if method == 'pointbiserialr':
            for i in range(self.attr_num):
                for j in range(i + 1, self.attr_num):
                    temp, _ = stats.pointbiserialr(self.dfattr.iloc[:, i], self.dfattr.iloc[:, j])
                    if temp > 0.8:
                        attr_relate.append((i, j, temp))
            return attr_relate

        pearson_corr_mat = self.dfattr.corr(method=method)

        for i in range(self.attr_num):
            for j in range(i + 1, self.attr_num):
                temp = pearson_corr_mat.iloc[i, j]
                if temp * temp > 0.64:
                    if flag > 0:
                        attr_relate.append((i, j, temp * temp))
                    elif temp > 0.8:
                        attr_relate.append((i, j, temp))
        return attr_relate

    def get_primary_components(self, ncomponents):
        """
        主成分分析法特征提取
        :param ncomponents: 目标维数
        :return: 将特征数据降维到目标维度后的dataframe
        """
        pca = PCA(n_components=ncomponents)
        return pca.fit_transform(self.valuearray).tolist()

    def get_dftarget_value_counts(self):
        return self.dftarget.value_counts().count()

    def get_machine_learning_algorithm(self, machine_learning_algorithm, fit_intercept, normalize, copy_X, n_jobs):
        mlal_dict = {}
        pandas.set_option('float_format', lambda x: '%.3f' % x)
        if machine_learning_algorithm == "LinearRegression":
            reg = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs).fit(
                self.dfattr, self.dftarget)
            temp = {}
            for i in range(len(reg.coef_)):
                temp[self.col_name[i]] = float(reg.coef_[i])
            mlal_dict['coef'] = temp
            mlal_dict['intercept_'] = reg.intercept_
            print(mlal_dict)
        return mlal_dict
