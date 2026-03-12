"use client"

import { useState, useMemo } from "react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { ArrowLeft, ExternalLink, Bookmark, User, BookOpen, Calendar, Filter, ChevronDown } from "lucide-react"
import Link from "next/link"

interface Paper {
  id: number
  title: string
  author: string
  year: string
  journal: string
  link: string
  citation: string
  image: string
}

const RESEARCH_CASES: Paper[] = [
  { id: 1, title: "A Topic Modeling Comparison Between LDA, NMF, Top2Vec, and BERTopic to Demystify Twitter Posts", author: "Egger R, Yu J", year: "2022", journal: "Frontiers in Sociology", link: "https://www.frontiersin.org/journals/sociology/articles/10.3389/fsoc.2022.886498/full", citation: "[1]Egger R, Yu J. A Topic Modeling Comparison Between LDA, NMF, Top2Vec, and BERTopic to Demystify Twitter Posts [J]. Frontiers in Sociology, 2022, 7.", image: "/papers/image1.png" },
  { id: 2, title: "Computational vs. qualitative: Analyzing different approaches in identifying networked frames during the Covid-19 Crisis", author: "Kermani H, Makou A B, Tafreshi A, 等", year: "2023", journal: "International Journal of Social Research Methodology", link: "https://www.tandfonline.com/doi/full/10.1080/13645579.2023.2186566", citation: "[2]Kermani H, et al. Computational vs. qualitative: Analyzing different approaches in identifying networked frames during the Covid-19 Crisis [J]. IJSRM, 2023, 27(4): 401–415.", image: "/papers/image2.png" },
  { id: 3, title: "AutoTM 2.0: Automatic Topic Modeling Framework for Documents Analysis", author: "Khodorchenko M, Butakov N, Zuev M, 等", year: "2024", journal: "arXiv", link: "https://arxiv.org/abs/2410.00655", citation: "[3]Khodorchenko M, et al. AutoTM 2.0: Automatic Topic Modeling Framework for Documents Analysis [Z]. arXiv, 2024.", image: "/papers/image3.png" },
  { id: 4, title: "Prompting Large Language Models for Topic Modeling", author: "Wang H, Prakash N, Hoang N K, 等", year: "2023", journal: "arXiv / IEEE", link: "https://ieeexplore.ieee.org/abstract/document/10386113", citation: "[4]Wang H, et al. Prompting Large Language Models for Topic Modeling [Z]. arXiv, 2023.", image: "/papers/image4.png" },
  { id: 5, title: "Enhancing Short-Text Topic Modeling with LLM-Driven Context Expansion and Prefix-Tuned VAEs", author: "Akash P S, Chang K C-C", year: "2024", journal: "arXiv", link: "https://arxiv.org/abs/2410.03071", citation: "[5]Akash P S, Chang K C-C. Enhancing Short-Text Topic Modeling with LLM-Driven Context Expansion and Prefix-Tuned VAEs [Z]. arXiv, 2024.", image: "/papers/image5.png" },
  { id: 6, title: "Topic research in fuzzy domain: Based on LDA topic Modelling", author: "Yu D, Fang A, Xu Z", year: "2023", journal: "Information Sciences", link: "https://www.sciencedirect.com/science/article/pii/S0020025523011854", citation: "[6]Yu D, Fang A, Xu Z. Topic research in fuzzy domain: Based on LDA topic Modelling [J]. Information Sciences, 2023, 648: 119600.", image: "/papers/image6.png" },
  { id: 7, title: "GOOD AND BAD SOCIOLOGY: DOES TOPIC MODELLING MAKE A DIFFERENCE?", author: "BARANOWSKI M, CICHOCKI P", year: "2021", journal: "Society Register", link: "https://pressto.amu.edu.pl/index.php/sr/article/view/31045", citation: "[7]BARANOWSKI M, CICHOCKI P. GOOD AND BAD SOCIOLOGY: DOES TOPIC MODELLING MAKE A DIFFERENCE? [J]. Society Register, 2021, 5(4): 7–22.", image: "/papers/image7.png" },
  { id: 8, title: "Exploring Trends in Environmental, Social, and Governance Themes and Their Sentimental Value Over Time", author: "Park J, Choi W, Jung S-U", year: "2022", journal: "Frontiers in Psychology", link: "https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2022.890435/full", citation: "[8]Park J, et al. Exploring Trends in ESG Themes and Their Sentimental Value Over Time [J]. Frontiers in Psychology, 2022, 13.", image: "/papers/image8.png" },
  { id: 9, title: "Identifying learners' topical interests from social media content to enrich their course preferences in MOOCs", author: "Zankadi H, Idrissi A, Daoudi N, 等", year: "2022", journal: "Education and Information Technologies", link: "https://link.springer.com/article/10.1007/s10639-022-11373-1", citation: "[9]Zankadi H, et al. Identifying learners' topical interests from social media content [J]. EIT, 2022, 28(5): 5567.", image: "/papers/image9.png" },
  { id: 10, title: "Differential exposure to drinking water contaminants in North Carolina: Evidence from structural topic modeling", author: "Sohns A", year: "2023", journal: "Journal of Environmental Management", link: "https://www.sciencedirect.com/science/article/pii/S0301479723003882", citation: "[10]Sohns A. Differential exposure to drinking water contaminants in North Carolina [J]. JEM, 2023, 336: 117600.", image: "/papers/image10.png" },
  { id: 11, title: "Contrastive learning for hierarchical topic Modeling", author: "Mao P, Chen H, Rao Y, 等", year: "2024", journal: "Natural Language Processing Journal", link: "https://www.sciencedirect.com/science/article/pii/S2949719124000062", citation: "[11]Mao P, et al. Contrastive learning for hierarchical topic Modeling [J]. NLPJ, 2024, 6: 100058.", image: "/papers/image11.png" },
  { id: 12, title: "Neural Topic Modeling via Discrete Variational Inference", author: "Gupta A, Zhang Z", year: "2023", journal: "ACM Trans. on Intelligent Systems and Technology", link: "https://dl.acm.org/doi/full/10.1145/3570509", citation: "[12]Gupta A, Zhang Z. Neural Topic Modeling via Discrete Variational Inference [J]. ACM TIST, 2023, 14(2): 1–33.", image: "/papers/image12.png" },
  { id: 13, title: "Topic Extraction: BERTopic's Insight into the 117th Congress's Twitterverse", author: "Mendonça M, Figueira Á", year: "2024", journal: "Informatics", link: "https://www.mdpi.com/2227-9709/11/1/8", citation: "[13]Mendonça M, Figueira Á. Topic Extraction: BERTopic's Insight into the 117th Congress's Twitterverse [J]. Informatics, 2024, 11(1): 8.", image: "/papers/image13.png" },
  { id: 14, title: "Beyond standardization: A comprehensive review of topic modeling validation methods for computational social science", author: "Bernhard-Harrer J, Ashour R, Eberl J-M, 等", year: "2025", journal: "Political Science Research and Methods", link: "https://www.cambridge.org/core/journals/political-science-research-and-methods/article/1D92CCD7C51491C9F6BE490BA5B434C4", citation: "[14]Bernhard-Harrer J, et al. Beyond standardization: A comprehensive review of topic modeling validation methods [J]. PSRM, 2025: 1–19.", image: "/papers/image14.png" },
  { id: 15, title: "A systematic review of the use of topic models for short text social media Analysis", author: "Laureate C D P, Buntine W, Linger H", year: "2023", journal: "Artificial Intelligence Review", link: "https://link.springer.com/article/10.1007/s10462-023-10471-x", citation: "[15]Laureate C D P, et al. A systematic review of the use of topic models for short text social media Analysis [J]. AIR, 2023, 56(12): 14223–14255.", image: "/papers/image15.png" },
  { id: 16, title: "Enhancing Social Media Content Analysis with Advanced Topic Modeling Techniques: A Comparative Study", author: "Nanayakkara A C, Thennakoon G A D M", year: "2024", journal: "ICTer", link: "https://icter.sljol.info/articles/10.4038/icter.v17i1.7276", citation: "[16]Nanayakkara A C, Thennakoon G A D M. Enhancing Social Media Content Analysis [J]. ICTer, 2024, 17(1): 40–47.", image: "/papers/image16.png" },
  { id: 17, title: "Exploring topic models to discern cyber threats on Twitter: A case study on Log4Shell", author: "Wang Y, Bashar M A, Chandramohan M, 等", year: "2023", journal: "Intelligent Systems with Applications", link: "https://www.sciencedirect.com/science/article/pii/S2667305323001059", citation: "[17]Wang Y, et al. Exploring topic models to discern cyber threats on Twitter [J]. ISA, 2023, 20: 200280.", image: "/papers/image17.png" },
  { id: 18, title: "Depression, anxiety, and burnout in academia: topic modeling of PubMed abstracts", author: "Lezhnina O", year: "2023", journal: "Frontiers in Research Metrics and Analytics", link: "https://www.frontiersin.org/journals/research-metrics-and-analytics/articles/10.3389/frma.2023.1271385/full", citation: "[18]Lezhnina O. Depression, anxiety, and burnout in academia: topic modeling of PubMed abstracts [J]. FRMA, 2023, 8.", image: "/papers/image18.png" },
  { id: 19, title: "An integrated view of Quantum Technology? Mapping Media, Business, and Policy Narratives", author: "Suter V, Ma C, Poehlmann G, 等", year: "2024", journal: "arXiv", link: "https://arxiv.org/abs/2408.02236", citation: "[19]Suter V, et al. An integrated view of Quantum Technology? Mapping Media, Business, and Policy Narratives [Z]. arXiv, 2024.", image: "/papers/image19.png" },
  { id: 20, title: "Temporal analysis of computational economics: A topic modeling Approach", author: "Mishra M, Vishwakarma S K, Malviya L, 等", year: "2024", journal: "Int. J. of System Assurance Eng. and Management", link: "#", citation: "[20]Mishra M, et al. Temporal analysis of computational economics: A topic modeling Approach [J]. IJSAEM, 2024.", image: "/papers/image20.png" },
  { id: 21, title: "Exploring three pillars of construction robotics via Dual-track quantitative Analysis", author: "Liu Y, Alias A H B, Haron N A, 等", year: "2024", journal: "Automation in Construction", link: "https://www.sciencedirect.com/science/article/pii/S0926580524001274", citation: "[21]Liu Y, et al. Exploring three pillars of construction robotics via Dual-track quantitative Analysis [J]. AiC, 2024, 162: 105391.", image: "/papers/image21.png" },
  { id: 22, title: "Efficient topic identification for urgent MOOC Forum posts using BERTopic and traditional topic modeling", author: "Khodeir N, Elghannam F", year: "2024", journal: "Education and Information Technologies", link: "https://link.springer.com/article/10.1007/s10639-024-13003-4", citation: "[22]Khodeir N, Elghannam F. Efficient topic identification for urgent MOOC Forum posts [J]. EIT, 2024, 30(5): 5501–5527.", image: "/papers/image22.png" },
  { id: 23, title: "Enhancing API Documentation through BERTopic Modeling and Summarization", author: "Naghshzan A, Ratte S", year: "2023", journal: "arXiv", link: "https://arxiv.org/abs/2308.09070", citation: "[23]Naghshzan A, Ratte S. Enhancing API Documentation through BERTopic Modeling and Summarization [Z]. arXiv, 2023.", image: "/papers/image23.png" },
  { id: 24, title: "Finding the structure of parliamentary motions in the Swedish Riksdag 1971–2015", author: "Bruinsma B, Johansson M", year: "2023", journal: "Quality & Quantity", link: "https://link.springer.com/article/10.1007/s11135-023-01802-9", citation: "[24]Bruinsma B, Johansson M. Finding the structure of parliamentary motions in the Swedish Riksdag [J]. Q&Q, 2023, 58(4): 3275–3301.", image: "/papers/image24.png" },
  { id: 25, title: "基于多维测度指标的技术主题识别与颠覆性特征演化研究——以光电子信息产业为例", author: "张肃, 蔡天勇", year: "2026", journal: "现代情报", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQz_mKALW5S1n0b5OMtqeMqY4tG7dFy7d0_1ec223XM5Y04cfVlCgiguE3WWyLDbSKPqONvvutsXq2MJRke4zLSpdwzaNwLMqhizCDyA6R0OFHNmiAZDHree7n3VY8PcC7fRVqQVLN_Gypsi1adAER7SLXY5dFScybk=&uniplatform=NZKPT", citation: "[1]张肃,蔡天勇. 基于多维测度指标的技术主题识别与颠覆性特征演化研究——以光电子信息产业为例[J/OL].现代情报,2026.", image: "/papers/image25.png" },
  { id: 26, title: "情感如何驱动认同？——基于主题模型与情感分析的\u201C网红\u201D文物传播机制研究", author: "赵琳, 曹文龙", year: "2026", journal: "新媒体与社会", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQyDFNP9A1PiAsiTcwjyVZt3f2WVs9p5qrfuqPF_3LsVfUXGoLiGRhKY6TELqhmBCaU9TLstN8AqrqRQdVaXLeduD7RFPU-YVINpRxJX73xeC-Gr1K6L9aB2GybFWekDmUpUKfW9ldpkMOvYknXOFkJJ3zpxMoJVTGI=&uniplatform=NZKPT", citation: "[2]赵琳,曹文龙. 情感如何驱动认同？——基于主题模型与情感分析的\u201C网红\u201D文物传播机制研究[J/OL].新媒体与社会,2026.", image: "/papers/image26.png" },
  { id: 27, title: "突发公共事件网络舆情的双重共情演化机制与引导策略研究——以Manner咖啡店泼咖啡粉事件为例", author: "相甍甍, 纪泽旭, 张柳, 等", year: "2026", journal: "情报理论与实践", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQwZ-YREpckRdUtB_QFSVyt2Hnk4nqqmTCBa5DCPyw7j_08mt1dErKflSp9QjaRJA6pYepPK0Vpzu1Zqux5LTjutbDLC9l-gSSyxc5dNoCdDN1bXwBUame4bk_cxOFkV30OgbUqyhgZgaqJZr5wPnD_FfO_Mn60wlFM=&uniplatform=NZKPT", citation: "[3]相甍甍,等. 突发公共事件网络舆情的双重共情演化机制与引导策略研究[J/OL].情报理论与实践,2026.", image: "/papers/image27.png" },
  { id: 28, title: "自然语言处理对政治学研究的方法改进及使用局限", author: "陈若凡", year: "2026", journal: "求索", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQxbzQuXhAEFFBdxSmiiTfw5nkRds413tH05GwrJnUgZIDWwOIVmp2OBSVwaPQHkFUWI6vF8F4JwWEMrT3KbHdtTx0FIByfmIpTJIc3Q_aeyAgRYMnFPL0BNxGy5kHndxmxjRhFnAFMJIvQom07AKmnf4LZqf8wO_0s=&uniplatform=NZKPT", citation: "[4]陈若凡. 自然语言处理对政治学研究的方法改进及使用局限[J/OL].求索,2026,(01):154-166.", image: "/papers/image28.png" },
  { id: 29, title: "三维框架下中美韩三国教师教育数字化转型政策比较研究——基于LDA主题模型和PMC指数模型", author: "钱小龙, 宋子昀, 黄蓓蓓", year: "2026", journal: "中国远程教育", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQwG4CRl8G7vRnc8HqM6jf2mBNdgdei5Jei9zYJkYwAck7pVExKtbPt3KiyyasTt5Z3kYlAUpsMPhf0YyTg-Bf6THALsfIUooINFv6z5rdCtuVPjgJ-A-TWRcWWAgtyX1Tz-NrMVF4Z0u0B4C0jplfAq&uniplatform=NZKPT", citation: "[5]钱小龙,等. 三维框架下中美韩三国教师教育数字化转型政策比较研究[J].中国远程教育,2026,46(01):143-168.", image: "/papers/image29.png" },
  { id: 30, title: "企业家精神对人工智能企业创新价值链的作用机制研究——基于科学—创业双元视角的多维分析", author: "邹家祎, 郭燕青, 张巧", year: "2026", journal: "科技进步与对策", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQxMuVkPg2pW1zroWK1xFn3QWkX1fXkxY07qcMfUwC3YTNTz2xVKUkG8Qver3Yfnr_OWy3XeeFWL1Hqq_ga4aYNKkAaKnGEM00zxcS1CbKr-PTwRk7Wfe3JmEKH03ofDRwnzzbD1B5GwwPPc4SYJCF2VgQO-V6-EsGo=&uniplatform=NZKPT", citation: "[6]邹家祎,等. 企业家精神对人工智能企业创新价值链的作用机制研究[J/OL].科技进步与对策,2026.", image: "/papers/image30.png" },
  { id: 31, title: "长江经济带省级氢能产业政策体系的量化评价研究——基于\u201C主题-工具-效力\u201D的三维框架分析", author: "顾东明, 张田飞, 李胜会", year: "2026", journal: "干旱区资源与环境", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQytCxxmjNXeHep6TxnZBpHgk6iYlilbgoOubaDq3HZrK9wcmVVRBH7kYyLnsLraBmcy5w3lzsdHpMjeQflEU-OkPv0YROdmnoGKbuspXtuob5BRNL5QWrMSdMwQpVzpfZKTnE1hl8EPIT8PUYPrnXhZ&uniplatform=NZKPT", citation: "[7]顾东明,等. 长江经济带省级氢能产业政策体系的量化评价研究[J].干旱区资源与环境,2026,40(02):10-20.", image: "/papers/image31.png" },
  { id: 32, title: "人工智能领域前沿主题的多维指标识别与动态演化分析", author: "胡泽文, 韩雅蓉, 张欣雨, 等", year: "2026", journal: "图书馆论坛", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQzcHFCLB06a4n2FQm7Efc4a6r9S0Z7Gwvb2iy9iB4cknlg_cZ6rb59nFbjhn6YJpskD25WpXnqZut8oKQgFo4WS9UNSdfHTml0kFnPZAsgursFlhrBUhTVnnTAuTeYSs5R6LA-CQlPTU1ntx3cPC5hHxGAMBWVUX6s=&uniplatform=NZKPT", citation: "[8]胡泽文,等. 人工智能领域前沿主题的多维指标识别与动态演化分析[J/OL].图书馆论坛,2026.", image: "/papers/image32.png" },
  { id: 33, title: "驱动新质生产力发展的政策组态及其演化轨迹——基于TJ-QCA的实证研究", author: "林艳, 敬燕妮, 孙云帆", year: "2025", journal: "管理学刊", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQypvTmcT0h3K-J6S3SM0Kdd7DsxIAns2g--jTx2adzImjOKEpIowPIYqRXPjBrudhVzmG9qFQiX7jadjCuZ1XowvgnH--SRiiXQi5u6X8t3hPd5RS7HVnAjduQZ4OdP8SQvvla2RSf2moe_k2fAoLj6&uniplatform=NZKPT", citation: "[9]林艳,等. 驱动新质生产力发展的政策组态及其演化轨迹[J].管理学刊,2025,38(06):63-79.", image: "/papers/image33.png" },
  { id: 34, title: "基于LDA主题建模的美国图书馆员AI能力需求探究与启示", author: "孙丽娟", year: "2025", journal: "图书与情报", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQx5UxRYvw1NOGpWLql1U_u8COixUuqHU5HMsWw_4YhW4gdaCB-jAlb1dVoNlRJSMedqV9c6WFfgo3066BtqP41Vsu2SS7a98z43aOMR1bJJoLWzvDGgd1CjdF6FWYFxw8_3V6V6PRlQ9AE_rrNtUZCzqPqJD2FWo_U=&uniplatform=NZKPT", citation: "[10]孙丽娟. 基于LDA主题建模的美国图书馆员AI能力需求探究与启示[J].图书与情报,2025,(06):104-112.", image: "/papers/image34.png" },
  { id: 35, title: "谁从应对气候变化中获益——气候治理受益主体的媒体叙事与话语变迁", author: "陈晓彤, 曾繁旭", year: "2025", journal: "新闻记者", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQx5UxRYvw1NOGpWLql1U_u8COixUuqHU5HMsWw_4YhW4gdaCB-jAlb1dVoNlRJSMedqV9c6WFfgo3066BtqP41Vsu2SS7a98z43aOMR1bJJoLWzvDGgd1CjdF6FWYFxw8_3V6V6PRlQ9AE_rrNtUZCzqPqJD2FWo_U=&uniplatform=NZKPT", citation: "[11]陈晓彤,曾繁旭. 谁从应对气候变化中获益——气候治理受益主体的媒体叙事与话语变迁[J].新闻记者,2025,(12):35-51.", image: "/papers/image35.png" },
  { id: 36, title: "基于动态主题模型的我国公共文化服务政策演进分析", author: "谢紫悦, 陈雅, 杜佳, 等", year: "2026", journal: "图书馆杂志", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQw-_1Of52I8Uj2Wq4oyUa_rIn4O6cNywoXBoxZPLelNGEjtPrO1krR5n-H1GN98pCt9u_jyTavKGXsCJkTqCuvlmotzYioQKBiDzbVZDXukDKGrkNfP5WNksHURpTOCBCEhdaLY72VC94ugWURtV4WXklkE6NtgbJI=&uniplatform=NZKPT", citation: "[12]谢紫悦,等. 基于动态主题模型的我国公共文化服务政策演进分析[J/OL].图书馆杂志,2026.", image: "/papers/image36.png" },
  { id: 37, title: "美国人工智能政策的技术民族主义属性与应对", author: "温婷, 姜南, 姜银鑫", year: "2026", journal: "科学学研究", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQwWd3KXofmanbKxxu8bCJZpBUjK9NhNUH1vmYY2d5ut-EUNyfNzA_3mrXcBkXcofu-3pW0QlXFTkiLTMdY3jWREFADSxaowoM0H1wyP3kc2aPSTw0aM66mXGHYFIbld2y_Rd6k39YeYLdFlHwD-eKjRWpvxDS-4x10=&uniplatform=NZKPT", citation: "[13]温婷,等. 美国人工智能政策的技术民族主义属性与应对[J/OL].科学学研究,2026.", image: "/papers/image37.png" },
  { id: 38, title: "中国教育专业学位案例的主题分析与建议——基于教育专业学位入库案例的混合研究", author: "吴立宝, 刘若凡", year: "2025", journal: "研究生教育研究", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQyxYHEKoQ0rXk-c0PRgukGO_X092eRZT-xPVsSo0P1p38gKLF33sg0O3-gH5cI-wj1XoqKE0SDX7baHBFKFV9dttCm2wt3tAnklT2UMlG6GYfCLFdrQHXudtQIQdo_5hg_0WG9M5Rh0tg70U2YP1HyD&uniplatform=NZKPT", citation: "[14]吴立宝,刘若凡. 中国教育专业学位案例的主题分析与建议[J].研究生教育研究,2025,(06):68-76.", image: "/papers/image38.png" },
  { id: 39, title: "梯度培育政策下专精特新企业内生发展动力机制研究", author: "谌飞龙, 王光华", year: "2025", journal: "当代财经", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQw29TP0aLw2fV7AZfaGBsLZ3jXivjuBtiI5ydaj0s3pfqClb-nMf4H4a6hJMc2WXrXB9TBAA8wgXSFxTQ7AyyA_tdsu9Ej9X_PAVkREfXrFagF19JTZZoV6uXUCisNi-MOu1LJBUCLQgDjId0jFVbqta4gZ7X-dOKg=&uniplatform=NZKPT", citation: "[15]谌飞龙,王光华. 梯度培育政策下专精特新企业内生发展动力机制研究[J].当代财经,2025,(12):114-128.", image: "/papers/image39.png" },
  { id: 40, title: "中国政府绩效管理自主知识体系的演变逻辑与实践进路", author: "包国宪, 钟小婷, 成杨影", year: "2025", journal: "中国行政管理", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQyC5A3nuAXwq-_vq4P9Ng3nDCzTFcaXg3-hC1iQV95Qs9o2eRDB3blYF6ClOhvuCxwcHG6d9MuIx_wI2JanKE0PuEOqKl4uPOMAxG02AX5MOOf65i2F5lmVsQ4Wa_1JPjj4lQ9SbRkSd_bKZ8yMeHhz&uniplatform=NZKPT", citation: "[16]包国宪,等. 中国政府绩效管理自主知识体系的演变逻辑与实践进路[J].中国行政管理,2025,41(11):98-113.", image: "/papers/image40.png" },
  { id: 41, title: "生育友好视域下我国托育政策的时空嬗变特征与优化向度——基于31省397份政策文本的LDA主题建模", author: "洪秀敏, 吕阳", year: "2025", journal: "人口与经济", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQwoiRGHYjJoqaPMjEHTgDI_UBWp8ZxbQdsl3iBojIioalBKD00Cila2Mp3Qy3MHSAu7BytOBp9U8fddWGotDhDMYE-OCzLwz5qESPDSw9GAH7962TmYGwiIVBFqAMYzyNjCtPTxBZ_FBee1P5KRvPQ5JeduEII5Gvs=&uniplatform=NZKPT", citation: "[17]洪秀敏,吕阳. 生育友好视域下我国托育政策的时空嬗变特征与优化向度[J].人口与经济,2025,(06):59-72.", image: "/papers/image41.png" },
  { id: 42, title: "颠覆性技术早期识别方法的新进展", author: "王政媛, 靳军宝, 郑玉荣, 等", year: "2025", journal: "科技管理研究", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQyDf03vfi4ij6WAOZdsF2lfNqo53ZVtE3evHTBdC6fz6Kfcxpvc8hBTL_-Cz9uD6dflYczP5n8Xmu_1DRqQWbaq58mQOuWCEQwDIi4ATev6XCjnKDSIe5CnTyr3nUyuzbFMqZGdLIVC7jru3W6dzUXB5tv12M8bM4o=&uniplatform=NZKPT", citation: "[18]王政媛,等. 颠覆性技术早期识别方法的新进展[J].科技管理研究,2025,45(22):1-12.", image: "/papers/image42.png" },
  { id: 43, title: "基于BERTOPIC模型的数据要素领域主题挖掘与内容分析", author: "杨智勇, 王慧", year: "2025", journal: "图书馆理论与实践", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQxxZKo4uzNwpNPH3kBqGf5jz_Fu69aOFCftXUAgDbyfQ3ADvyflj-eSBGSr9wkzxVa7VdNl9hakbRIz431j-7mQ4z3e7kbwHeXi3HmOGYsjiaFx7dk5V249x8aCEzsDwn1xsFcggALQwqg9eOwyz4LwkiMukwT0b_o=&uniplatform=NZKPT", citation: "[19]杨智勇,王慧. 基于BERTOPIC模型的数据要素领域主题挖掘与内容分析[J].图书馆理论与实践,2025,(06):78-89+126.", image: "/papers/image43.png" },
  { id: 44, title: "基于BERT主题模型的\u300A西游记\u300B英、日译本读者体验对比研究", author: "马爽, 毛文伟", year: "2025", journal: "外语教学", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQyxdt0b_swd8Kbocll1RLpigV8e_ErPhe0JhEjL-svL8KuaYVXiaOk-hbJkwCHzL5cO7qESEQi9XkPuQndp_InULDTn4GyG9B3BRjUZXwNSkPFO_ym--lnn8tj-cF1E4r4Iybaf_Tt6Do73EReVt4nfLhrEqnAIuno=&uniplatform=NZKPT", citation: "[20]马爽,毛文伟. 基于BERT主题模型的\u300A西游记\u300B英、日译本读者体验对比研究[J].外语教学,2025,46(06):72-79.", image: "/papers/image44.png" },
  { id: 45, title: "探寻颠覆性技术的基础研究来源：基于个体创新主体视角", author: "何郁冰, 何丽, 徐美娟", year: "2025", journal: "情报学报", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQygzxyLtnXn_wB-l8A_u_sFv199vL48K6g2o8UCvnNE3sl1BIFbFb6vkpUxwrheS2c8Ar_tRVDigl0l9PPUhUnOFgNLW6JH4i1gw-kkMPXFiukjfoscrJgvt2wtdB9zvs1Ufz9k-HA6L0bzLeYmztFYCbdaiUo_Fa0=&uniplatform=NZKPT", citation: "[21]何郁冰,等. 探寻颠覆性技术的基础研究来源：基于个体创新主体视角[J].情报学报,2025,44(10):1300-1314.", image: "/papers/image45.png" },
  { id: 46, title: "基于技术预见视角的弱信号识别研究综述", author: "曹海艳, 王暖臣, 穆歌, 等", year: "2025", journal: "情报学报", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQxXoQLQw1jpBpl6VF14N551A9PWgd6PX48PugfyB0e-73ytnZAKyxjB7paHN6CsNpBQBoQ3_6fW5Fs9zWnbF1tdSCBgvHm9V7PhG4drgApzmlGTwqeRumOuzl1rF2rAu4mJys-ceuibaiWxCJ3k-XCfLAOS1VsbNi8=&uniplatform=NZKPT", citation: "[22]曹海艳,等. 基于技术预见视角的弱信号识别研究综述[J].情报学报,2025,44(10):1342-1358.", image: "/papers/image46.png" },
  { id: 47, title: "AI赋能信息资源管理学科研究方法现状、问题与趋势", author: "马捷, 顾英驰, 蒲思彤", year: "2025", journal: "图书情报工作", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQwBerJ_yoBU-OMC1U2uzpTPTCD8nspmglJRq4dsNHb0HYsZISl91WNhrsXSsGUGjc4jzDgWlRaTYIanB-VoSn8w4hjYyzM5lHSBbXLDN6Z16J6vMfuudh8JY_fE5l9RNt596e-L26gMs01sbO1zS-yrUs2kwSGPeqs=&uniplatform=NZKPT", citation: "[23]马捷,等. AI赋能信息资源管理学科研究方法现状、问题与趋势[J].图书情报工作,2025,69(20):16-29.", image: "/papers/image47.png" },
  { id: 48, title: "基于\u201C力度-主题-结构\u201D联动视角的科技金融政策文本量化研究", author: "喻平, 熊王百卉", year: "2025", journal: "科技管理研究", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQyWTeCb3GxQhxia-_GkheCA6GXoLtvlmGfchJFzYM8xrA6nHIAWdKUfqUXRJ4Ol5BI0Tgl1lB2CVG-hfvRZzTMe_b7c1lJOz3vW1xtVciKmeFcBHS0QT7gChil5f3NMktWwlwPqfxntfXwhGqcIgo0hGDjBG3s-DMk=&uniplatform=NZKPT", citation: "[24]喻平,熊王百卉. 基于\u201C力度-主题-结构\u201D联动视角的科技金融政策文本量化研究[J].科技管理研究,2025,45(20):22-34.", image: "/papers/image48.png" },
  { id: 49, title: "跨区域技术转移政策演进研究——基于京津冀协同发展十年的变迁", author: "李子彪, 姚菲芸", year: "2025", journal: "科学学与科学技术管理", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQyJXd3V_qOI1WeEK3SBcKi1UAY6YmpciLBc_ed7I_fkGlagjHWjsC8GzwdwZtDsN3A3la-jLrtI0iDkEIqy2Eg2tskcLaAUHY5ngINndfYbP3n06V9-UYzYi3QoAuRtj8Q_-iRgOgxycOkR3tH2mZL2Ov1y23x5sLM=&uniplatform=NZKPT", citation: "[25]李子彪,姚菲芸. 跨区域技术转移政策演进研究[J/OL].科学学与科学技术管理,2025.", image: "/papers/image49.png" },
  { id: 50, title: "基于大语言模型的网络舆情事理图谱构建与演化分析——以体育赛事为例", author: "姜帆, 郭顺利", year: "2026", journal: "情报科学", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQz50VDvVyQu7uibYnl9B0ufRwcCRZCqWzjt36jIwVyYwKcfGUGuNCy78W6cDrcU5oIrGgTO6s_Jg1X21dSXC97b9y_jS9izDiK42hnUgYPBEtDbxNfdS-bI09CL2Be2Begy4ZwdmEfUMJ0r-3_YFVRSTWRiifkJQJ0=&uniplatform=NZKPT", citation: "[26]姜帆,郭顺利. 基于大语言模型的网络舆情事理图谱构建与演化分析[J/OL].情报科学,2026.", image: "/papers/image50.png" },
  { id: 51, title: "基于TopicGPT模型的智慧养老研究主题挖掘与演化分析", author: "曾鹏翔, 刘天畅, 蒲政同, 等", year: "2026", journal: "情报科学", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQwdYZjock3_7npxqHRITw25FxKyn888O7blozx3u1WqOSxt6hhF8UK5zWcPQeSWFyrgiAsCD0HqvEmS2TkjYoXDW25zmYM1AH0d-1PPWdSD8HPQen0n9GFo32VLqrD2nHF4YvJZZz_00FA-HwZRWEO1HhO18rHS9Fg=&uniplatform=NZKPT", citation: "[27]曾鹏翔,等. 基于TopicGPT模型的智慧养老研究主题挖掘与演化分析[J/OL].情报科学,2026.", image: "/papers/image51.png" },
  { id: 52, title: "基于因果图的研究前沿演化动因识别研究", author: "白如江, 任前前, 陈鑫, 等", year: "2026", journal: "现代情报", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQyNNcpbXN1zcmaeTinY5Vr4oPdrVMrXUe0l75HYE3qJPceOO0wHcT0J3x-hPcrpJg2fAk0_cl1d96b_5dMI2Ili1qY0NzypPXVCKvNnsDO27EPVaHHPL1SF080F-2ayHN9eRx9h9PARr3iqVemLoPwJ&uniplatform=NZKPT", citation: "[28]白如江,等. 基于因果图的研究前沿演化动因识别研究[J].现代情报,2026,46(02):45-60.", image: "/papers/image52.png" },
  { id: 53, title: "粤港澳大湾区国际形象构建研究——基于1309篇国际新闻报道的实证分析", author: "朱颖, 朱梅基, 邓伟健", year: "2026", journal: "新媒体与社会", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQwQWfwTkJDzD3KrNz87lQNeOqm10Xccszv4VcdyuUufjI72LIM5U3df5_UBCi-WZu-AIVi4E6GxegkD2ecRcc3GmKWWUEY2Fdlk9aTebStV0KvdGm2gmVzew-qX4Vx-q9SQKjrVEsXvcJewVw_GnRw2SriaKMwTq30=&uniplatform=NZKPT", citation: "[29]朱颖,等. 粤港澳大湾区国际形象构建研究——基于1309篇国际新闻报道的实证分析[J/OL].新媒体与社会,2026.", image: "/papers/image53.png" },
  { id: 54, title: "市场导向下数字赋能绿色创新体系构建——多政策文本扎根与机器学习聚类研究", author: "谢吉青, 周昱希, 谢家平", year: "2025", journal: "上海财经大学学报", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQx28G6CzeqIyr1AyQxujiGrLxzGwyU65RU29e7dQUOgW1zZHu27Dj2qzCb9HAAKKotOKtT0EQS_v7gvCtuAG9AEdm7QyUC91ZLmHWXtgfMF0RJYg8651cU5oJDl_v7f36EF26IXKYwvlshwOOtQ3tC4-oxbspuJfM0=&uniplatform=NZKPT", citation: "[30]谢吉青,等. 市场导向下数字赋能绿色创新体系构建[J].上海财经大学学报,2025,27(05):108-122.", image: "/papers/image54.png" },
  { id: 55, title: "中国旅游用地政策历史演进及变迁逻辑——基于\u201C主体-工具-主题\u201D三维框架", author: "张赛楠, 宋昌耀, 厉新建", year: "2026", journal: "旅游学刊", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQyJzJJb90DUYpdjXsThZM6t2LSCcDFOFW2Yub9RpOovgeJIgQelobmObvinQ1yITQiGF03fx25-cqMQUunI5XUeGoaa2ABVO93uXHn1JH1rOgFAxPnNatI4G2r-tbyCFY422Gv0U2rJIn96ax-bc6742MkyTiiERdg=&uniplatform=NZKPT", citation: "[31]张赛楠,等. 中国旅游用地政策历史演进及变迁逻辑[J/OL].旅游学刊,2026.", image: "/papers/image55.png" },
  { id: 56, title: "\u201C环境\u201D与\u201C地缘政治\u201D的叙事博弈：国际报道如何影响中国电动汽车海外销量", author: "杨帆, 张鑫", year: "2025", journal: "新闻与传播研究", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQx02fiO7qwzCLE0bfrsxoK006ofwU9Qn7ThCeyl1kxRt2sgaEFSgAJDvL2IIrCIoCOfNKefDa15qRJ4GNpWldLy0NLiFx84GS96mlNJUGwRimhtyHIXtT8Q5rqJ1W_bS7894PLVht86V5d3buDHfC6oHMcIjDfY87A=&uniplatform=NZKPT", citation: "[32]杨帆,张鑫. \u201C环境\u201D与\u201C地缘政治\u201D的叙事博弈：国际报道如何影响中国电动汽车海外销量[J].新闻与传播研究,2025,32(09):111-125+128.", image: "/papers/image56.png" },
  { id: 57, title: "压力型体制下政府重视与科技创新水平提升——基于2013—2023年省级\u300A政府工作报告\u300B的文本分析", author: "汤峰, 杨雪冬", year: "2025", journal: "社会科学", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQwzxiBTVhdKqpxHrQtgO8IoAExhhSjnZZj_GEYXbd43DobpvJStHFFgr00ITZXz5kmmOnTEzq5g8BaVtfRhs872nkqiauXvfrMuOzQJoDSg2jPM9HHolRQGS9d8VkGwEn_OS7S_ypedvR2KY_S6ZP_LKf7pe290PSE=&uniplatform=NZKPT", citation: "[33]汤峰,杨雪冬. 压力型体制下政府重视与科技创新水平提升[J].社会科学,2025,(09):97-111.", image: "/papers/image57.png" },
  { id: 58, title: "元宇宙教育应用技术与工程教育深度学习：技术主题识别与应用场景对接", author: "范惠明, 金莹", year: "2025", journal: "电化教育研究", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQwMYBpiIrPc3TirvrzPskoY6dPf_5dMKPAbLNOqJbO_6ISwHt4aDP_2Q2Ly_18TtvpvsCxKVZsiOQuuNjpxQN4XJ8gPojaHwvlfwVah6g-xNnpjFfowUNon7aYiXd0a388kT87ioZ21f6mLcdXzHRedPFRxrTbnp8Y=&uniplatform=NZKPT", citation: "[34]范惠明,金莹. 元宇宙教育应用技术与工程教育深度学习[J].电化教育研究,2025,46(10):55-62.", image: "/papers/image58.png" },
  { id: 59, title: "基于改进词移嵌入的文本表示方法研究", author: "岑咏华, 李文敬, 刘贤祖", year: "2025", journal: "情报学报", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQwZbW6dukgBXb7gnBnG1LfmfZhhjYJpIhAYQ8TK8qHGdUeQfCNmWf2kw-pGrwoLXgkMZ7vry0qSL3aUZMOGfz3G-KBxBwpntSCy6T7-bJp3F0n3SKC8cZQgFnb1B42aEvS4Lkjv-OS8JmilPcwQn_bvfj5Nexoq15Q=&uniplatform=NZKPT", citation: "[35]岑咏华,等. 基于改进词移嵌入的文本表示方法研究[J].情报学报,2025,44(09):1173-1191.", image: "/papers/image59.png" },
  { id: 60, title: "以球为媒：\u201C苏超\u201D体育赛事赋能城市品牌传播的实践探索研究", author: "蔡馥谣, 金书颖", year: "2026", journal: "新媒体与社会", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQzDpOi-uNO5Futu3PCIdjWJSE6JO322G23QCisvMEOzM3XM-9hyiooGa8qPXS25wHj35IbZZl1xu7NcPNUcfURn62MLhBSqcQfnYdmN1LFlPL3nhQNG0JNcPONZcmshnQsRuAUp57RUaSCi7K3el5aL8q47qdn3lak=&uniplatform=NZKPT", citation: "[36]蔡馥谣,金书颖. 以球为媒：\u201C苏超\u201D体育赛事赋能城市品牌传播的实践探索研究[J/OL].新媒体与社会,2026.", image: "/papers/image60.png" },
  { id: 61, title: "地质公园游客感知意象的主题识别与空间结构——以陕西翠华山国家地质公园为例", author: "李显正, 张大钊, 赵振斌, 等", year: "2025", journal: "干旱区资源与环境", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQyvPiLgbBsoU_1WG3Z3aJebxb2vRkVJNZW81RF_k69ZjO_aOi8evdysGxWIzXZlfuMxNHrPl1wSUk2DFzAoJ6U46HXTjZ1Jplt9pfwWVbvC00CmnWfeoTvESHRQQhjPNUTes8SvfvRUiwoxFSeL7eNvr15LNqcMduQ=&uniplatform=NZKPT", citation: "[37]李显正,等. 地质公园游客感知意象的主题识别与空间结构[J].干旱区资源与环境,2025,39(10):188-198.", image: "/papers/image61.png" },
  { id: 62, title: "国家公园生态系统文化服务时空格局及影响因素研究——以雅鲁藏布大峡谷为例", author: "张曼迪, 范梦余, 王学峰, 等", year: "2025", journal: "干旱区资源与环境", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQzkMWkeMY0YhoWqlkkLf_Of7KmrUpeV45N-GNaIigdFhopbo0Ln1x3N4BBEKqFLfUc7KVTFWqc_JraojuK3yUVMl4UST0vgIAXmakH3HIZPceRZcQ9aMN1qDRP0zFNbWc7jFJeiU2FHJZmKSNQVumdLaw9-jaWMabg=&uniplatform=NZKPT", citation: "[38]张曼迪,等. 国家公园生态系统文化服务时空格局及影响因素研究[J].干旱区资源与环境,2025,39(10):103-113.", image: "/papers/image62.png" },
  { id: 63, title: "融合文本和图像的个性化需求预测方法——基于有限偏好视角", author: "姜元春, 李怡, 钱洋, 等", year: "2025", journal: "管理科学学报", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQxKqOgcYitdSepNxZrZa_E5JdmErK5JJgN7CRhb90-3zgS43wVwTR49s0IDC2E4O5RGoPPsHON0OwL00z_1NaPXWOXVSfF2dXx__lwjeBIaaNNh6Ix4UqZfUFvjWsLpaDKwAoRPgSTuJoM5bQYnKz9VaT63FyAFQQo=&uniplatform=NZKPT", citation: "[39]姜元春,等. 融合文本和图像的个性化需求预测方法[J].管理科学学报,2025,28(09):52-64.", image: "/papers/image63.png" },
  { id: 64, title: "基于BERTopic和长短期记忆网络模型的我国生成式人工智能技术演进、发展趋势与创新生态分析", author: "黄欢, 王楚尧", year: "2025", journal: "科技管理研究", link: "https://kns.cnki.net/kcms2/article/abstract?v=MXvIvFkaDQykq8InWcnlqsC3ynSiEe1X6BQATbOLINVVPXKPICYvOL1gl1Uk5hV55kBU1g8cFWnHwX1Fr6Yl9g2kFp_yFapQR9NGoEB6o6ZO27rIpqRlJ4s8kUPerNd_OvqCrPCm9G49ICmdoiRKHjaCuiQNLw84XKnVc4wafRI=&uniplatform=NZKPT", citation: "[40]黄欢,王楚尧. 基于BERTopic和长短期记忆网络模型的我国生成式人工智能技术演进分析[J].科技管理研究,2025,45(17):178-190.", image: "/papers/image64.png" },
]

const YEAR_FILTERS = ["全部", "2026", "2025", "2024", "2023", "2022", "2021"]
const LANG_FILTERS = ["全部", "中文", "英文"]

function isChineseTitle(title: string) {
  return /[\u4e00-\u9fa5]/.test(title)
}

export default function CasesPage() {
  const [yearFilter, setYearFilter] = useState("全部")
  const [langFilter, setLangFilter] = useState("全部")

  const filteredPapers = useMemo(() => {
    return RESEARCH_CASES.filter((p) => {
      if (yearFilter !== "全部" && p.year !== yearFilter) return false
      if (langFilter === "中文" && !isChineseTitle(p.title)) return false
      if (langFilter === "英文" && isChineseTitle(p.title)) return false
      return true
    })
  }, [yearFilter, langFilter])

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-50">
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-20 -left-20 w-80 h-80 bg-blue-200/30 rounded-full blur-3xl" />
        <div className="absolute top-40 -right-20 w-96 h-96 bg-purple-200/20 rounded-full blur-3xl" />
        <div className="absolute bottom-20 left-1/3 w-72 h-72 bg-indigo-200/20 rounded-full blur-3xl" />
      </div>

      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200/60">
        <div className="max-w-7xl mx-auto px-5 sm:px-6 h-14 flex items-center justify-between">
          <Link href="/">
            <Button variant="ghost" size="sm" className="gap-2 text-slate-600 hover:text-slate-900">
              <ArrowLeft className="w-4 h-4" />
              返回首页
            </Button>
          </Link>
          <div className="flex items-center gap-2">
            <img src="/theta-logo.png" alt="THETA" className="h-7 w-auto" />
          </div>
        </div>
      </header>

      <section className="max-w-7xl mx-auto px-5 sm:px-6 pt-12 pb-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-blue-50 text-blue-600 rounded-full text-sm font-medium mb-6">
            <Bookmark className="w-4 h-4" />
            学术资源
          </div>
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-black text-slate-900 tracking-tight mb-4">
            学术<span className="text-blue-600">案例库</span>
          </h1>
          <p className="text-slate-600 text-lg max-w-2xl mx-auto mb-8">
            探索使用主题模型进行研究的优秀学术论文，获取灵感和方法论参考
          </p>

          <div className="flex flex-wrap items-center justify-center gap-3 mb-2">
            <div className="flex items-center gap-1.5">
              <Filter className="w-4 h-4 text-slate-400" />
              <span className="text-sm text-slate-500">筛选：</span>
            </div>
            <div className="flex gap-1.5 flex-wrap">
              {YEAR_FILTERS.map((y) => (
                <button
                  key={y}
                  onClick={() => setYearFilter(y)}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
                    yearFilter === y
                      ? "bg-blue-600 text-white shadow-sm"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  {y}
                </button>
              ))}
            </div>
            <div className="w-px h-5 bg-slate-200 hidden sm:block" />
            <div className="flex gap-1.5">
              {LANG_FILTERS.map((l) => (
                <button
                  key={l}
                  onClick={() => setLangFilter(l)}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
                    langFilter === l
                      ? "bg-indigo-600 text-white shadow-sm"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  {l}
                </button>
              ))}
            </div>
            <span className="text-xs text-slate-400 ml-2">
              共 {filteredPapers.length} 篇
            </span>
          </div>
        </motion.div>
      </section>

      <section className="max-w-7xl mx-auto px-5 sm:px-6 pb-20">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
        >
          {filteredPapers.map((paper, index) => (
            <PaperCard key={paper.id} paper={paper} index={index} />
          ))}
        </motion.div>

        {filteredPapers.length === 0 && (
          <div className="text-center py-20">
            <BookOpen className="w-12 h-12 text-slate-300 mx-auto mb-4" />
            <p className="text-slate-500">暂无匹配的论文</p>
          </div>
        )}

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-12 text-center"
        >
          <p className="text-sm text-slate-400 mb-4">
            共收录 {RESEARCH_CASES.length} 篇学术论文，持续更新中...
          </p>
          <Link href="/">
            <Button variant="outline" className="gap-2">
              <ArrowLeft className="w-4 h-4" />
              返回首页
            </Button>
          </Link>
        </motion.div>
      </section>
    </div>
  )
}

function PaperCard({ paper, index }: { paper: Paper; index: number }) {
  const isChinese = isChineseTitle(paper.title)

  return (
    <motion.div
      initial={{ opacity: 0, y: 30, rotateY: -5 }}
      animate={{ opacity: 1, y: 0, rotateY: 0 }}
      transition={{ duration: 0.5, delay: index * 0.03 }}
      className="paper-card-flip h-[380px]"
    >
      <div className="paper-card-flip-inner">
        {/* ========== 正面：论文首页截图 ========== */}
        <div className="paper-card-front rounded-2xl border-2 border-slate-200/80 bg-white overflow-hidden shadow-lg hover:shadow-2xl transition-shadow duration-300">
          <div className="relative w-full h-full">
            <img
              src={paper.image}
              alt={paper.title}
              className="w-full h-full object-cover object-top"
              onError={(e) => {
                const target = e.target as HTMLImageElement
                target.style.display = "none"
                target.nextElementSibling?.classList.remove("hidden")
              }}
            />
            <div className="hidden absolute inset-0 bg-gradient-to-br from-slate-100 to-blue-50 flex items-center justify-center p-6">
              <div className="text-center">
                <BookOpen className="w-10 h-10 text-slate-300 mx-auto mb-3" />
                <p className="text-sm text-slate-500 line-clamp-3 font-medium">{paper.title}</p>
              </div>
            </div>

            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent p-4 pt-12">
              <h3 className="text-white text-sm font-bold line-clamp-2 leading-snug mb-1 drop-shadow-lg">
                {paper.title}
              </h3>
              <p className="text-white/80 text-xs line-clamp-1">{paper.author}</p>
            </div>

            <div className="absolute top-3 left-3 flex gap-1.5">
              <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold shadow-sm ${
                isChinese
                  ? "bg-red-500/90 text-white"
                  : "bg-blue-500/90 text-white"
              }`}>
                {isChinese ? "中文" : "EN"}
              </span>
              <span className="px-2 py-0.5 bg-black/50 backdrop-blur-sm rounded-md text-[10px] font-bold text-white">
                {paper.year}
              </span>
            </div>

            <div className="absolute bottom-3 right-3 px-2.5 py-1 bg-white/20 backdrop-blur-sm rounded-full text-[10px] text-white/90 font-medium opacity-0 group-hover:opacity-100 transition-opacity">
              悬停查看详情 ↻
            </div>
          </div>
        </div>

        {/* ========== 背面：详细信息 ========== */}
        <div className={`paper-card-back rounded-2xl border-2 border-slate-200/80 overflow-hidden shadow-lg hover:shadow-2xl transition-shadow duration-300 ${
          isChinese
            ? "bg-gradient-to-br from-slate-800 via-red-950 to-slate-900"
            : "bg-gradient-to-br from-slate-800 via-blue-950 to-slate-900"
        }`}>
          <div
            className="absolute inset-0 opacity-10"
            style={{
              backgroundImage: `
                linear-gradient(to right, white 1px, transparent 1px),
                linear-gradient(to bottom, white 1px, transparent 1px)
              `,
              backgroundSize: "12px 12px",
            }}
          />

          <div className="relative h-full p-5 flex flex-col text-white">
            <div className="flex items-start gap-2 mb-1">
              <span className={`shrink-0 px-2 py-0.5 rounded text-[10px] font-bold ${
                isChinese ? "bg-red-500/30 text-red-300" : "bg-blue-500/30 text-blue-300"
              }`}>
                {isChinese ? "中文" : "EN"}
              </span>
              <span className="px-2 py-0.5 bg-white/10 rounded text-[10px] font-bold text-white/70">
                {paper.year}
              </span>
            </div>

            <h3 className="font-bold text-sm leading-snug mb-3 mt-2 line-clamp-4">
              {paper.title}
            </h3>

            <div className="space-y-2.5 mb-4">
              <div className="flex items-start gap-2">
                <User className="w-3.5 h-3.5 mt-0.5 shrink-0 text-white/60" />
                <p className="text-xs text-white/80 line-clamp-2">{paper.author}</p>
              </div>
              <div className="flex items-start gap-2">
                <BookOpen className="w-3.5 h-3.5 mt-0.5 shrink-0 text-white/60" />
                <p className="text-xs text-white/80 line-clamp-1 italic">{paper.journal}</p>
              </div>
            </div>

            <div className="mb-4 pl-3 border-l-2 border-white/20">
              <p className="text-[11px] text-white/60 leading-relaxed line-clamp-4">
                {paper.citation}
              </p>
            </div>

            <div className="mt-auto pt-3 border-t border-white/20">
              {paper.link && paper.link !== "#" ? (
                <a
                  href={paper.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded-lg text-xs font-medium text-white/90 transition-colors"
                  onClick={(e) => e.stopPropagation()}
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  查看原文
                </a>
              ) : (
                <span className="text-[11px] text-white/40">暂无在线链接</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
