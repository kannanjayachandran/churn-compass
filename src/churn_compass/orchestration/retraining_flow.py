"""
Churn Compass - Automated Retraining Flow

Conditional retraining based on drift detection with gated approval.
"""

import argparse
from typing import Optional, Dict
from datetime import datetime, timezone

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from churn_compass import settings, setup_logger
from churn_compass.monitoring import monitoring_flow
from churn_compass.orchestration import training_flow

logger = setup_logger(__name__)


@task(name="evaluate_retraining_criteria")
def evaluate_retraining_criteria(
    monitoring_results: Dict, 
    drift_threshold: float = 0.5, 
    performance_drop_threshold: float = 0.05
) -> Dict:
    """
    Docstring for evaluate_retraining_criteria
    
    :param monitoring_results: Description
    :type monitoring_results: Dict
    :param drift_threshold: Description
    :type drift_threshold: float
    :param performance_drop_threshold: Description
    :type performance_drop_threshold: float
    :return: Description
    :rtype: Dict[Any, Any]
    """
    logger.info("Evaluate retraining criteria")

    data_drift = monitoring_results.get('data_drift', {})
    performance_drift = monitoring_results.get('performance_drift', {})
    alerts = monitoring_results.get('alerts', [])

    reasons = []
    should_retrain = False

    # check data drift
    if data_drift.get('drift_detected'):
        drift_share = data_drift.get('drift_share', 0)
        if drift_share >= drift_threshold:
            should_retrain = True
            reasons.append(f"Significant data drift detected: {drift_share:.1%} of features")
    
    # check performance degradation
    if performance_drift and performance_drift.get('performance_degraded'):
        drops = performance_drift.get('metric_drops', {})
        pr_auc_drop = drops.get('pr_auc_drop', 0)

        if pr_auc_drop >= performance_drop_threshold:
            should_retrain = True
            reasons.append(f"Performance degradation: PR-AUC dropped by {pr_auc_drop:.3f}")
    
    # Check for critical alerts
    critical_alerts = [a for a in alerts if a['severity'] == 'critical']
    if critical_alerts:
        should_retrain = True
        reasons.append(f"{len(critical_alerts)} critical alerts triggered")
    
    decision = {
        'should_retrain': should_retrain, 
        'reasons': reasons, 
        'drift_share': data_drift.get('drift_share', 0), 
        'performance_drop': performance_drift.get('metric_drops', {}).get('pr_auc_drop', 0) if performance_drift else 0, 
        'critical_alerts': len(critical_alerts), 
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Retraining criteria evaluated", 
        extra={
            'should_retrain': should_retrain, 
            'reasons_count': len(reasons), 
        }
    )

    return decision

@task(name="check_manual_approval")
def check_manual_approval(
    decision: Dict, 
    require_approval: bool = True
) -> bool:
    """
    Docstring for check_manual_approval
    
    :param decision: Description
    :type decision: Dict
    :param require_approval: Description
    :type require_approval: bool
    :return: Description
    :rtype: bool
    """
    if not require_approval:
        logger.info("Manual approval not required, proceeding with retraining")
        return True
    
    if not decision['should_retrain']:
        logger.info("Retraining not recommended, skipping")
        return False
    
    # In production, this would check for approval from:
    # - Human-in-the-loop system
    # - Approval queue in database
    # - External approval API

    # For now, check settings flag
    approval_granted = settings.auto_retrain_enabled

    logger.info(
        f"Manual approval check: {"APPROVED" if approval_granted else "PENDING"}", 
        extra={
            "require_approval": require_approval, 
            "approved": approval_granted
        }
    )
    
    return approval_granted

@task(name="create_retraining_summary")
def create_retraining_summary(
    decision: Dict,
    approved: bool,
    training_results: Optional[Dict] = None
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    markdown = f"""
# Churn Compass Retraining Report

**Generated**: {timestamp}

## Retraining Decision

**Recommendation**: {'🔴 RETRAIN' if decision['should_retrain'] else '🟢 NO RETRAIN'}
**Approval Status**: {'✅ APPROVED' if approved else '⏳ PENDING / ❌ DENIED'}

### Reasons
"""
    if decision['reasons']:
        for r in decision['reasons']:
            markdown += f"- {r}\n"
    else:
        markdown += "- All metrics within acceptable ranges\n"

    markdown += f"""
### Metrics
- **Data Drift Share**: {decision['drift_share']:.1%}
- **PR-AUC Drop**: {decision['performance_drop']:.4f}
- **Critical Alerts**: {decision['critical_alerts']}
"""

    if training_results:
        optimized = training_results["optimized_results"]
        test_metrics = optimized["metrics"]["test"]

        markdown += f"""
## Retraining Results

**Status**: ✅ COMPLETED  
**Run ID**: `{optimized['run_id']}`

### Test Performance
- **PR-AUC**: {test_metrics['pr_auc']:.4f}
- **Precision@{settings.top_k_percent:.0%}**: {test_metrics['topk_precision_at_k']:.4f}
- **Recall@{settings.top_k_percent:.0%}**: {test_metrics['topk_recall_at_k']:.4f}
- **Lift@{settings.top_k_percent:.0%}**: {test_metrics['topk_lift_at_k']:.2f}x

**Model Registered**: {'Yes' if training_results['model_registered'] else 'No'}
"""
    elif decision['should_retrain']:
        markdown += "\n## Retraining Results\n\n**Status**: ⏸️ Awaiting Approval\n"
    else:
        markdown += "\n## Retraining Results\n\n**Status**: ⏭️ Skipped\n"

    markdown += f"\n---\n*Report generated at {timestamp}*"
    return markdown

@flow(
    name="automated_retraining", 
    description="Conditional retraining based on drift detection", 
    log_prints=True
)
def retraining_flow(
    reference_data_path: str, 
    current_data_source: str, 
    training_data_path: str, 
    current_data_type: str = "parquet", 
    check_performance: bool = True, 
    require_approval: bool = True, 
    drift_threshold: float = 0.5, 
    performance_threshold: float = 0.05, 
    n_trials: int = 50
) -> Dict: 
    """
    Docstring for retraining_flow
    
    :param reference_data_path: Description
    :type reference_data_path: str
    :param current_data_source: Description
    :type current_data_source: str
    :param training_data_path: Description
    :type training_data_path: str
    :param current_data_type: Description
    :type current_data_type: str
    :param check_performance: Description
    :type check_performance: bool
    :param require_approval: Description
    :type require_approval: bool
    :param drift_threshold: Description
    :type drift_threshold: float
    :param performance_threshold: Description
    :type performance_threshold: float
    :param n_trials: Description
    :type n_trials: int
    :return: Description
    :rtype: Dict[Any, Any]
    """
    logger.info(
        "Starting automated retraining flow", 
        extra={
            "require_approval": require_approval, 
            "drift_threshold": drift_threshold, 
        }
    )

    # 1. Run monitoring
    print("\n" + "="*80)
    print("Step 1: Running Monitoring")
    print("="*80 + "\n")
    
    monitoring_results = monitoring_flow(
        reference_path=reference_data_path, 
        current_source=current_data_source, 
        current_type=current_data_type, 
        check_performance=check_performance
    )
   
    # Step 2: Evaluate criteria
    print("\n" + "="*80)
    print("Step 2: Evaluating Retraining Criteria")
    print("="*80 + "\n")
    
    decision = evaluate_retraining_criteria(
        monitoring_results,
        drift_threshold=drift_threshold,
        performance_drop_threshold=performance_threshold
    )
    
    print(f"Retraining Recommended: {decision['should_retrain']}")
    if decision['reasons']:
        print("Reasons:")
        for reason in decision['reasons']:
            print(f"  - {reason}")
    
    # Step 3: Check approval
    approved = check_manual_approval(decision, require_approval)
    
    # Step 4: Retrain if approved
    training_results = None
    
    if approved and decision['should_retrain']:
        print("\n" + "="*80)
        print("Step 3: Retraining Model")
        print("="*80 + "\n")
        
        training_results = training_flow(
            data_path=training_data_path,
            skip_tuning=False,
            n_trials=n_trials,
            register_model=True
        )
    elif decision['should_retrain']:
        print("\n⏸️  Retraining recommended but awaiting approval")
    else:
        print("\n✅ No retraining needed - all metrics within acceptable ranges")
    
    # Step 5: Create summary
    summary_markdown = create_retraining_summary(
        decision,
        approved,
        training_results
    )
    
    # Create Prefect artifact
    create_markdown_artifact(
        key="retraining-report",
        markdown=summary_markdown,
        description="Automated retraining decision and results"
    )
    
    # Print summary
    print("\n" + "="*80)
    print(summary_markdown)
    print("="*80 + "\n")
    
    results = {
        'monitoring_results': monitoring_results,
        'decision': decision,
        'approved': approved,
        'training_results': training_results,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    logger.info(
        "Retraining flow completed",
        extra={
            'retrained': training_results is not None,
            'approval_required': require_approval,
            'approved': approved
        }
    )
    
    return results


def main():
    """CLI entry point for retraining flow."""
    parser = argparse.ArgumentParser(description="Automated retraining flow")
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Reference data for monitoring"
    )
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Current production data"
    )
    parser.add_argument(
        "--training-data",
        type=str,
        required=True,
        help="Data for retraining"
    )
    parser.add_argument(
        "--current-type",
        type=str,
        default="parquet",
        choices=["parquet", "csv", "sql"],
        help="Type of current data"
    )
    parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Skip manual approval requirement"
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.5,
        help="Data drift threshold"
    )
    parser.add_argument(
        "--performance-threshold",
        type=float,
        default=0.05,
        help="Performance drop threshold"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Optuna trials"
    )
    
    args = parser.parse_args()
    
    try:
        results = retraining_flow(
            reference_data_path=args.reference,
            current_data_source=args.current,
            training_data_path=args.training_data,
            current_data_type=args.current_type,
            require_approval=not args.no_approval,
            drift_threshold=args.drift_threshold,
            performance_threshold=args.performance_threshold,
            n_trials=args.trials
        )
        
        if results['training_results']:
            print(f"✅ Retraining completed. Run ID: {results['training_results']['final_run_id']}")
        elif results['decision']['should_retrain']:
            print("⏸️  Retraining recommended but not approved")
        else:
            print("✅ No retraining needed")
    
    except Exception as e:
        logger.error("Retraining flow failed", exc_info=True)
        print(f"❌ Retraining flow failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()