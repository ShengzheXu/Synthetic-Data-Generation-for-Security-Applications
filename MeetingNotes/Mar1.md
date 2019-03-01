# Mar 1 Meeting Keynote

## Again about baseline1&2

    Material
    1 IP with 1342589 rows;
    10 IPs with 1342589 rows;
    10 IPs with 1342589 * 10 rows;

    Experiment Plan
    1. Make data for 1 IP, with expansion log(#B), delta time, log(#B-1)
    2. combine 10 IPs to one data
    3. Input: 1 IP/10 IPs => baseline 1 => Output: gen_1IP_baseline1/gen_10IPs_baseline1
    4. Fix BN baseline2, with log(B-1) col.
    Then input: 1IP/10IPs => baseline2 => output: gen_1IP/10IPs_baseline2
    5. conduct k-l divergence and NMI on 2*2 results (1IP/10IPs * bl1/bl2), comparing with real data.

## Summary of new materials