import os
import time
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import random
from datetime import datetime
from tqdm import tqdm

load_dotenv(verbose=True)

DATASETS_CUSTOM_PATH = os.getenv("DATASETS_CUSTOM_PATH")
if not DATASETS_CUSTOM_PATH:
    raise ValueError("DATASETS_CUSTOM_PATH 환경 변수가 설정되지 않았습니다!")


def generate_random_ip():
    """임의의 IPv4 주소를 생성합니다."""
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def generate_netflow_dataset(n_samples=50000, random_state=42):
    """
    NetFlow v3 피처를 가진 임의의 데이터셋을 생성합니다.

    Parameters:
    -----------
    n_samples : int, default=50000
        생성할 데이터의 개수
    random_state : int, default=42
        랜덤 시드

    Returns:
    --------
    pandas.DataFrame
        NetFlow 피처를 가진 데이터셋
    """
    np.random.seed(random_state)
    random.seed(random_state)

    # 전체 진행 상황을 위한 메인 진행바
    main_pbar = tqdm(
        total=100, desc="전체 데이터셋 생성", unit="%", position=0, leave=True
    )

    # 기본 타임스탬프 생성 (최근 1년 내)
    main_pbar.set_description("타임스탬프 생성 중")
    base_time = int(datetime.now().timestamp() * 1000)
    start_times = base_time - np.random.randint(0, 365 * 24 * 60 * 60 * 1000, n_samples)
    main_pbar.update(2)

    data = {}

    # IP 주소 및 포트
    main_pbar.set_description("IP 주소 생성 중 (소스)")
    data["IPV4_SRC_ADDR"] = [
        generate_random_ip()
        for _ in tqdm(range(n_samples), desc="소스 IP 생성", leave=False, position=1)
    ]
    main_pbar.update(2)

    main_pbar.set_description("IP 주소 생성 중 (목적지)")
    data["IPV4_DST_ADDR"] = [
        generate_random_ip()
        for _ in tqdm(range(n_samples), desc="목적지 IP 생성", leave=False, position=1)
    ]
    main_pbar.update(2)

    main_pbar.set_description("포트 번호 생성 중")
    data["L4_SRC_PORT"] = np.random.randint(1024, 65535, n_samples).astype(np.int64)
    data["L4_DST_PORT"] = np.random.choice(
        [80, 443, 22, 53, 25, 21, 23, 993, 995], size=n_samples
    ).astype(np.int64)
    main_pbar.update(2)

    # 프로토콜
    main_pbar.set_description("프로토콜 정보 생성 중")
    protocols = np.random.choice(
        [6, 17, 1], size=n_samples, p=[0.7, 0.25, 0.05]
    )  # TCP, UDP, ICMP
    data["PROTOCOL"] = protocols.astype(np.int64)
    data["L7_PROTO"] = np.random.choice(
        [1.0, 6.0, 17.0, 80.0, 443.0, 53.0], size=n_samples
    ).astype(np.float64)
    main_pbar.update(3)

    # 바이트 및 패킷 수
    main_pbar.set_description("바이트 및 패킷 수 생성 중")
    in_bytes = np.random.lognormal(mean=10, sigma=2, size=n_samples).astype(np.int64)
    out_bytes = np.random.lognormal(mean=9, sigma=2, size=n_samples).astype(np.int64)
    data["IN_BYTES"] = in_bytes
    data["OUT_BYTES"] = out_bytes
    data["IN_PKTS"] = np.random.poisson(lam=20, size=n_samples).astype(np.int64)
    data["OUT_PKTS"] = np.random.poisson(lam=15, size=n_samples).astype(np.int64)
    main_pbar.update(3)

    # 플로우 지속 시간
    main_pbar.set_description("플로우 지속 시간 생성 중")
    flow_duration = np.random.exponential(scale=1000, size=n_samples).astype(np.int64)
    data["FLOW_DURATION_MILLISECONDS"] = flow_duration
    data["DURATION_IN"] = (
        flow_duration * np.random.uniform(0.3, 0.7, n_samples)
    ).astype(np.int64)
    data["DURATION_OUT"] = (
        flow_duration * np.random.uniform(0.3, 0.7, n_samples)
    ).astype(np.int64)

    # 타임스탬프
    data["FLOW_START_MILLISECONDS"] = start_times.astype(np.int64)
    data["FLOW_END_MILLISECONDS"] = (start_times + flow_duration).astype(np.int64)
    main_pbar.update(3)

    # TCP 플래그
    main_pbar.set_description("TCP 플래그 생성 중")
    tcp_flags = np.random.randint(0, 8191, n_samples).astype(
        np.int64
    )  # TCP 플래그 조합
    data["TCP_FLAGS"] = tcp_flags
    data["CLIENT_TCP_FLAGS"] = (
        tcp_flags * np.random.uniform(0.4, 0.6, n_samples)
    ).astype(np.int64)
    data["SERVER_TCP_FLAGS"] = (
        tcp_flags * np.random.uniform(0.4, 0.6, n_samples)
    ).astype(np.int64)
    main_pbar.update(3)

    # TTL
    main_pbar.set_description("TTL 정보 생성 중")
    data["MIN_TTL"] = np.random.choice([64, 128, 255], size=n_samples).astype(np.int64)
    data["MAX_TTL"] = data["MIN_TTL"] + np.random.randint(0, 10, n_samples).astype(
        np.int64
    )

    # 패킷 길이
    min_pkt_len = np.random.randint(40, 1500, n_samples).astype(np.int64)
    max_pkt_len = min_pkt_len + np.random.randint(0, 500, n_samples).astype(np.int64)
    data["LONGEST_FLOW_PKT"] = max_pkt_len
    data["SHORTEST_FLOW_PKT"] = min_pkt_len
    data["MIN_IP_PKT_LEN"] = min_pkt_len
    data["MAX_IP_PKT_LEN"] = max_pkt_len
    main_pbar.update(3)

    # 바이트/초
    main_pbar.set_description("바이트/초 계산 중")
    data["SRC_TO_DST_SECOND_BYTES"] = (
        in_bytes / np.maximum(flow_duration / 1000.0, 0.001)
    ).astype(np.float64)
    data["DST_TO_SRC_SECOND_BYTES"] = (
        out_bytes / np.maximum(flow_duration / 1000.0, 0.001)
    ).astype(np.float64)
    main_pbar.update(3)

    # 재전송
    main_pbar.set_description("재전송 정보 생성 중")
    retransmitted_in_pkts = np.random.poisson(lam=1, size=n_samples).astype(np.int64)
    data["RETRANSMITTED_IN_PKTS"] = retransmitted_in_pkts
    data["RETRANSMITTED_IN_BYTES"] = (
        retransmitted_in_pkts * np.random.randint(64, 1500, n_samples)
    ).astype(np.int64)
    retransmitted_out_pkts = np.random.poisson(lam=1, size=n_samples).astype(np.int64)
    data["RETRANSMITTED_OUT_PKTS"] = retransmitted_out_pkts
    data["RETRANSMITTED_OUT_BYTES"] = (
        retransmitted_out_pkts * np.random.randint(64, 1500, n_samples)
    ).astype(np.int64)
    main_pbar.update(3)

    # 처리량
    main_pbar.set_description("처리량 계산 중")
    data["SRC_TO_DST_AVG_THROUGHPUT"] = (data["SRC_TO_DST_SECOND_BYTES"] * 8).astype(
        np.int64
    )
    data["DST_TO_SRC_AVG_THROUGHPUT"] = (data["DST_TO_SRC_SECOND_BYTES"] * 8).astype(
        np.int64
    )
    main_pbar.update(3)

    # 패킷 크기 분포
    main_pbar.set_description("패킷 크기 분포 생성 중")
    total_pkts = data["IN_PKTS"] + data["OUT_PKTS"]
    pkt_size_dist = np.random.dirichlet([5, 4, 3, 2, 1], size=n_samples)
    data["NUM_PKTS_UP_TO_128_BYTES"] = (total_pkts * pkt_size_dist[:, 0]).astype(
        np.int64
    )
    data["NUM_PKTS_128_TO_256_BYTES"] = (total_pkts * pkt_size_dist[:, 1]).astype(
        np.int64
    )
    data["NUM_PKTS_256_TO_512_BYTES"] = (total_pkts * pkt_size_dist[:, 2]).astype(
        np.int64
    )
    data["NUM_PKTS_512_TO_1024_BYTES"] = (total_pkts * pkt_size_dist[:, 3]).astype(
        np.int64
    )
    data["NUM_PKTS_1024_TO_1514_BYTES"] = (total_pkts * pkt_size_dist[:, 4]).astype(
        np.int64
    )
    main_pbar.update(3)

    # TCP 윈도우
    main_pbar.set_description("TCP 윈도우 생성 중")
    data["TCP_WIN_MAX_IN"] = np.random.choice(
        [65535, 8192, 16384, 32768], size=n_samples
    ).astype(np.int64)
    data["TCP_WIN_MAX_OUT"] = np.random.choice(
        [65535, 8192, 16384, 32768], size=n_samples
    ).astype(np.int64)
    main_pbar.update(3)

    # ICMP
    main_pbar.set_description("ICMP 정보 생성 중")
    icmp_types = np.random.choice(
        [0, 8, 3, 11], size=n_samples
    )  # Echo Reply, Echo Request, Destination Unreachable, Time Exceeded
    data["ICMP_IPV4_TYPE"] = icmp_types.astype(np.int64)
    data["ICMP_TYPE"] = (icmp_types * 256 + np.random.randint(0, 16, n_samples)).astype(
        np.int64
    )
    main_pbar.update(3)

    # DNS
    main_pbar.set_description("DNS 정보 생성 중")
    data["DNS_QUERY_ID"] = np.random.randint(1, 65535, n_samples).astype(np.int64)
    data["DNS_QUERY_TYPE"] = np.random.choice([1, 2, 5, 15, 28], size=n_samples).astype(
        np.int64
    )  # A, NS, CNAME, MX, AAAA
    data["DNS_TTL_ANSWER"] = np.random.choice(
        [300, 600, 3600, 86400], size=n_samples
    ).astype(np.int64)
    main_pbar.update(3)

    # FTP
    main_pbar.set_description("FTP 정보 생성 중")
    data["FTP_COMMAND_RET_CODE"] = np.random.choice(
        [200, 220, 230, 331, 530], size=n_samples
    ).astype(np.int64)
    main_pbar.update(3)

    # IAT (Inter-Arrival Time) - src to dst
    main_pbar.set_description("IAT 정보 생성 중 (Src->Dst)")
    iat_s2d_min = np.random.exponential(scale=10, size=n_samples).astype(np.int64)
    iat_s2d_max = iat_s2d_min + np.random.exponential(scale=100, size=n_samples).astype(
        np.int64
    )
    iat_s2d_avg = (
        (iat_s2d_min + iat_s2d_max) / 2 + np.random.normal(0, 5, n_samples)
    ).astype(np.int64)
    data["SRC_TO_DST_IAT_MIN"] = iat_s2d_min
    data["SRC_TO_DST_IAT_MAX"] = iat_s2d_max
    data["SRC_TO_DST_IAT_AVG"] = np.maximum(iat_s2d_avg, 1).astype(np.int64)
    data["SRC_TO_DST_IAT_STDDEV"] = np.random.exponential(
        scale=5, size=n_samples
    ).astype(np.int64)
    main_pbar.update(3)

    # IAT - dst to src
    main_pbar.set_description("IAT 정보 생성 중 (Dst->Src)")
    iat_d2s_min = np.random.exponential(scale=10, size=n_samples).astype(np.int64)
    iat_d2s_max = iat_d2s_min + np.random.exponential(scale=100, size=n_samples).astype(
        np.int64
    )
    iat_d2s_avg = (
        (iat_d2s_min + iat_d2s_max) / 2 + np.random.normal(0, 5, n_samples)
    ).astype(np.int64)
    data["DST_TO_SRC_IAT_MIN"] = iat_d2s_min
    data["DST_TO_SRC_IAT_MAX"] = iat_d2s_max
    data["DST_TO_SRC_IAT_AVG"] = np.maximum(iat_d2s_avg, 1).astype(np.int64)
    data["DST_TO_SRC_IAT_STDDEV"] = np.random.exponential(
        scale=5, size=n_samples
    ).astype(np.int64)
    main_pbar.update(3)

    # Label (이진 분류: 0=정상, 1=공격)
    main_pbar.set_description("레이블 생성 중")
    data["Label"] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]).astype(
        np.int64
    )
    main_pbar.update(3)

    # Attack 타입
    main_pbar.set_description("Attack 타입 생성 중")
    attack_types = [
        "Normal",
        "DoS",
        "Exploits",
        "Fuzzers",
        "Reconnaissance",
        "Worms",
        "Shellcode",
        "Backdoor",
        "Analysis",
    ]
    data["Attack"] = [
        attack_types[0] if label == 0 else random.choice(attack_types[1:])
        for label in tqdm(
            data["Label"],
            desc="Attack 타입 할당",
            leave=False,
            position=1,
            total=n_samples,
        )
    ]
    main_pbar.update(3)

    # DataFrame 생성 및 컬럼 순서 정렬
    main_pbar.set_description("DataFrame 생성 중")
    df = pd.DataFrame(data)
    main_pbar.update(5)

    # 컬럼 순서를 dataset.py의 dtypes 순서와 맞춤
    main_pbar.set_description("컬럼 순서 정렬 중")
    column_order = [
        "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS",
        "IPV4_SRC_ADDR",
        "L4_SRC_PORT",
        "IPV4_DST_ADDR",
        "L4_DST_PORT",
        "PROTOCOL",
        "L7_PROTO",
        "IN_BYTES",
        "IN_PKTS",
        "OUT_BYTES",
        "OUT_PKTS",
        "TCP_FLAGS",
        "CLIENT_TCP_FLAGS",
        "SERVER_TCP_FLAGS",
        "FLOW_DURATION_MILLISECONDS",
        "DURATION_IN",
        "DURATION_OUT",
        "MIN_TTL",
        "MAX_TTL",
        "LONGEST_FLOW_PKT",
        "SHORTEST_FLOW_PKT",
        "MIN_IP_PKT_LEN",
        "MAX_IP_PKT_LEN",
        "SRC_TO_DST_SECOND_BYTES",
        "DST_TO_SRC_SECOND_BYTES",
        "RETRANSMITTED_IN_BYTES",
        "RETRANSMITTED_IN_PKTS",
        "RETRANSMITTED_OUT_BYTES",
        "RETRANSMITTED_OUT_PKTS",
        "SRC_TO_DST_AVG_THROUGHPUT",
        "DST_TO_SRC_AVG_THROUGHPUT",
        "NUM_PKTS_UP_TO_128_BYTES",
        "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES",
        "TCP_WIN_MAX_IN",
        "TCP_WIN_MAX_OUT",
        "ICMP_TYPE",
        "ICMP_IPV4_TYPE",
        "DNS_QUERY_ID",
        "DNS_QUERY_TYPE",
        "DNS_TTL_ANSWER",
        "FTP_COMMAND_RET_CODE",
        "SRC_TO_DST_IAT_MIN",
        "SRC_TO_DST_IAT_MAX",
        "SRC_TO_DST_IAT_AVG",
        "SRC_TO_DST_IAT_STDDEV",
        "DST_TO_SRC_IAT_MIN",
        "DST_TO_SRC_IAT_MAX",
        "DST_TO_SRC_IAT_AVG",
        "DST_TO_SRC_IAT_STDDEV",
        "Label",
        "Attack",
    ]

    df = df[column_order]
    main_pbar.update(5)
    main_pbar.set_description("데이터셋 생성 완료")
    main_pbar.close()

    return df


if __name__ == "__main__":
    sel_n_samples = 500000
    output_path = os.path.join(
        DATASETS_CUSTOM_PATH,
        f"{sel_n_samples}s-NF-custom-dataset-{int(time.time())}.csv",
    )

    # 데이터셋 생성
    print(f"임의의 NetFlow 데이터셋 생성 중... 샘플 수: {sel_n_samples:,}개")
    print("=" * 60)
    start_time = time.time()

    generated_df = generate_netflow_dataset(n_samples=sel_n_samples, random_state=42)

    elapsed_time = time.time() - start_time
    print("=" * 60)
    print(f"\n✓ 데이터셋 생성 완료 (소요 시간: {elapsed_time:.2f}초)")

    print("\n생성된 데이터셋 정보:")
    print(f"- 행 개수: {len(generated_df):,}")
    print(f"- 열 개수: {len(generated_df.columns)}")
    print("\n컬럼 목록:")
    print(generated_df.columns.tolist())
    print("\nLabel 분포:")
    print(generated_df["Label"].value_counts())
    print("\nAttack 타입 분포:")
    print(generated_df["Attack"].value_counts())
    print("\n데이터 타입:")
    print(generated_df.dtypes)
    print("\n첫 5행 미리보기:")
    print(generated_df.head())

    # CSV로 저장
    print("\n파일 저장 중...")
    save_start = time.time()
    generated_df.to_csv(output_path, index=False)
    save_time = time.time() - save_start
    print(
        f"✓ 데이터셋이 '{output_path}'에 저장되었습니다. (소요 시간: {save_time:.2f}초)"
    )
