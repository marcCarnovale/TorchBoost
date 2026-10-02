from torchboost.adaptive.architecture_search import (
    AdmissionDecision,
    ArchitectureArchive,
    ArchitectureCandidate,
    ArchitectureCoordinates,
    ArchitectureEvidence,
    SuccessiveHalvingAllocator,
    TrainingBudget,
)


def ev(sel, nll, auc, params=100, examples=1000):
    return ArchitectureEvidence(
        selection_nll=sel,
        ranking_nll=nll,
        ranking_auc=auc,
        trainable_parameters=params,
        budget=TrainingBudget(
            optimizer_updates=10,
            examples_seen=examples,
            full_data_passes=1.0,
        ),
    )


def test_coordinates_are_multiaxis_and_mutable_by_copy():
    base=ArchitectureCoordinates(
        composition="latent",
        tree_depth=0,
        tree_count=0,
        layer_count=5,
        hidden_width=300,
        routing_hardness="hard",
        routing_geometry="axis",
        packet_type="affine",
        construction="gradient",
        aggregation="sequential",
    )
    grown=base.mutate(
        tree_depth=1,
        routing_hardness="soft",
        routing_geometry="oblique",
        global_polish=True,
    )
    assert base.tree_depth==0
    assert grown.tree_depth==1
    assert grown.routing_geometry=="oblique"
    assert grown.global_polish


def test_admission_requires_matched_heldout_nonregression_and_gain():
    archive=ArchitectureArchive()
    anchor=ArchitectureCandidate("anchor",ArchitectureCoordinates())
    parent=ArchitectureCandidate(
        "parent",ArchitectureCoordinates(),parent_id="anchor",mutation="continuation"
    )
    control=ArchitectureCandidate(
        "control",ArchitectureCoordinates(),parent_id="parent",mutation="continuation",
        evidence=ev(.57,.571,.768)
    )
    improved=ArchitectureCandidate(
        "grown",ArchitectureCoordinates(tree_depth=1),parent_id="parent",
        mutation="grow_zero_residual",evidence=ev(.569,.570,.769)
    )
    archive.add(anchor);archive.add(parent);archive.add(control);archive.add(improved)
    decision=archive.compare_to_control("grown","control")
    assert isinstance(decision,AdmissionDecision)
    assert decision.admitted
    assert decision.selection_delta<0
    assert decision.ranking_nll_delta<0
    assert decision.ranking_auc_delta>0


def test_admission_rejects_ranking_regression_even_if_selection_improves():
    archive=ArchitectureArchive()
    archive.add(ArchitectureCandidate("anchor",ArchitectureCoordinates()))
    archive.add(ArchitectureCandidate(
        "parent",ArchitectureCoordinates(),parent_id="anchor",mutation="continuation"
    ))
    archive.add(ArchitectureCandidate(
        "control",ArchitectureCoordinates(),parent_id="parent",
        mutation="continuation",evidence=ev(.57,.571,.768)
    ))
    archive.add(ArchitectureCandidate(
        "candidate",ArchitectureCoordinates(tree_depth=1),parent_id="parent",
        mutation="grow",evidence=ev(.568,.575,.767)
    ))
    assert not archive.compare_to_control("candidate","control").admitted


def test_admission_rejects_unmatched_budget():
    archive=ArchitectureArchive()
    archive.add(ArchitectureCandidate("anchor",ArchitectureCoordinates()))
    archive.add(ArchitectureCandidate(
        "parent",ArchitectureCoordinates(),parent_id="anchor",mutation="continuation"
    ))
    archive.add(ArchitectureCandidate(
        "control",ArchitectureCoordinates(),parent_id="parent",
        mutation="continuation",evidence=ev(.57,.571,.768,examples=1000)
    ))
    archive.add(ArchitectureCandidate(
        "candidate",ArchitectureCoordinates(tree_depth=1),parent_id="parent",
        mutation="grow",evidence=ev(.569,.570,.769,examples=2000)
    ))
    try:
        archive.compare_to_control("candidate","control")
    except ValueError as exc:
        assert "budgets are not matched" in str(exc)
    else:
        raise AssertionError("unmatched exposure must not be comparable")


def test_pareto_archive_keeps_quality_efficiency_tradeoffs():
    archive=ArchitectureArchive()
    a=ArchitectureCandidate("a",ArchitectureCoordinates(),evidence=ev(.57,.57,.77,100,1000))
    b=ArchitectureCandidate("b",ArchitectureCoordinates(),evidence=ev(.57,.569,.771,200,1000))
    c=ArchitectureCandidate("c",ArchitectureCoordinates(),evidence=ev(.57,.58,.75,300,1500))
    archive.add(a);archive.add(b);archive.add(c)
    assert set(archive.non_dominated())=={"a","b"}


def test_successive_halving_prefers_selection_then_ranking():
    rows=[
        ArchitectureCandidate("a",ArchitectureCoordinates(),evidence=ev(.570,.570,.770)),
        ArchitectureCandidate("b",ArchitectureCoordinates(),evidence=ev(.568,.580,.760)),
        ArchitectureCandidate("c",ArchitectureCoordinates(),evidence=ev(.569,.569,.771)),
        ArchitectureCandidate("d",ArchitectureCoordinates(),evidence=ev(.575,.560,.780)),
    ]
    allocator=SuccessiveHalvingAllocator(keep_fraction=.5)
    assert allocator.survivors(rows)==["b","c"]
